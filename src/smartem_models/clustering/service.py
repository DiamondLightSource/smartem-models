import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from smartem_backend.model.database import GridSquare, QualityPredictionModelParameter
from smartem_backend.mq_publisher import publish_gridsquare_model_prediction, publish_model_parameter_update
from smartem_backend.utils import setup_postgres_connection
from sqlmodel import Session, and_, func, select
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from smartem_models.clustering.calldata import SquareDataset, prepare_image
from smartem_models.clustering.grid_clustering import train
from smartem_models.clustering.models import EIAE
from smartem_models.parameter_models import InferenceParameters, InitParameters, UpdateParameters
from smartem_models.utils import read_img
from smartem_models.utils.parameter_updating_cluster import init_distributions, score, update_distribution

model_name = "dae-square"


def _set_model_parameters(
    dists: np.array,
    coords: list[tuple[str, tuple[float, float], int]],
    grid_uuid: str,
) -> None:
    for i, d in enumerate(dists):
        for j in range(len(d)):
            publish_model_parameter_update(
                grid_uuid=grid_uuid,
                model_name=model_name,
                key=str(j),
                value=float(d[j]),
                group=f"dist:{i}",
            )
    for c in coords:
        publish_model_parameter_update(
            grid_uuid=grid_uuid,
            model_name=model_name,
            key="x",
            value=float(c[1][0]),
            group=f"coordinates:{c[0]}",
        )
        publish_model_parameter_update(
            grid_uuid=grid_uuid,
            model_name=model_name,
            key="y",
            value=float(c[1][1]),
            group=f"coordinates:{c[0]}",
        )
        publish_model_parameter_update(
            grid_uuid=grid_uuid,
            model_name=model_name,
            key=str(c[0]),
            value=float(c[2]),
            group="cluster_indices",
        )
    return None


def initialise(params: InitParameters) -> None:
    engine = setup_postgres_connection()
    with Session(engine) as session:
        grid_squares = session.exec(select(GridSquare).where(GridSquare.grid_uuid == params.grid_uuid)).all()
    if not all(gs.image_path for gs in grid_squares):
        return None
    square_imgs = {i: Path(gs.image_path) for i, gs in enumerate(grid_squares)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_x = SquareDataset(square_imgs, transform=transforms.Resize(params.input_dim[-1], antialias=True))
    train_dataloader = DataLoader(train_x, batch_size=params.batch_size, shuffle=True, pin_memory=True)

    torch.set_num_threads(params.num_threads)
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)

    model = EIAE(
        alpha=torch.Tensor([[1.0, 1.0]]).to(device),
        input_dims=params.input_dim,
        hidden_dims=params.hidden_dims,
        lat_dim=params.latent_space_dim,
    )
    if params.input_model:
        model.load_state_dict(torch.load(params.input_model, map_location=device))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate, betas=(0.9, 0.999))

    losses = []
    epoch_losses: list[float] = [10**15]

    model.train()
    for _epoch in range(params.num_epochs):
        loss_record, epoch_loss = train(train_dataloader, model, optimizer, device)
        losses += loss_record
        epoch_losses.append(epoch_loss)

    model.eval()

    latent_coords = {}
    for sample in train_dataloader:
        coords = model.encode(Variable(sample["x1"]).to(device))[0].detach().cpu().numpy()
        for i, label in enumerate(sample["lab"].detach().cpu().numpy().flatten()):
            latent_coords[grid_squares[label].uuid] = coords[i]

    num_clusters = len(latent_coords) // 5
    labelled_coords = list(latent_coords.items())
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(
        np.array([p[1] for p in labelled_coords])
    )
    labelled_coords = [(p[0], p[1], q) for p, q in zip(labelled_coords, kmeans.labels_, strict=False)]
    hist = [[0.5 for p in labelled_coords if p[2] == label] for label in range(num_clusters)]
    largest_cluster = np.max([len(p) for p in hist])
    hist = [np.pad(p, (0, largest_cluster - len(p)), "constant", constant_values=(np.nan, np.nan)) for p in hist]
    init_dists = init_distributions(hist)

    if params.model_output_path:
        torch.save(model.state_dict(), params.model_output_path)
    if params.kmeans_output_path:
        with open(params.kmeans_output_path, "wb") as pkl:
            pickle.dump(kmeans, pkl)

    _set_model_parameters(init_dists, labelled_coords, params.grid_uuid)

    return None


def _add_cluster_index(grid_uuid: str, cluster_index: int, gridsquare: str, coords: np.array) -> None:
    publish_model_parameter_update(
        grid_uuid=grid_uuid,
        model_name=model_name,
        key="x",
        value=coords[0],
        group=f"coordinates:{gridsquare}",
    )
    publish_model_parameter_update(
        grid_uuid=grid_uuid,
        model_name=model_name,
        key="y",
        value=coords[1],
        group=f"coordinates:{gridsquare}",
    )
    publish_model_parameter_update(
        grid_uuid=grid_uuid,
        model_name=model_name,
        key=gridsquare,
        value=cluster_index,
        group="cluster_indices",
    )
    return None


def infer(params: InferenceParameters):
    torch.set_num_threads(params.num_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EIAE(
        alpha=torch.Tensor([[1.0, 1.0]]).to(device),
        input_dims=params.input_dim,
        hidden_dims=params.hidden_dims,
        lat_dim=params.latent_space_dim,
    )
    model.load_state_dict(torch.load(params.model_path, weights_only=True, map_location=device))
    model.eval()
    img = transforms.Resize(params.input_dim[-1], antialias=True)(prepare_image(read_img(params.gridsquare_img_path)))
    coords = model(img.unsqueeze(0))[2].detach().cpu().numpy()
    with open(params.kmeans_path, "rb") as pkl:
        kmeans = pickle.load(pkl)
    cluster_index = kmeans.predict([coords])
    _add_cluster_index(params.grid_uuid, cluster_index, params.gridsquare_uuid, coords)
    return coords


def _get_dist(grid_uuid: str, cluster_index: int, num_steps: int = 10) -> np.array:
    engine = setup_postgres_connection()
    with Session(engine) as session:
        if (
            session.exec(
                select(func.count(QualityPredictionModelParameter.key)).where(
                    QualityPredictionModelParameter.grid_uuid == grid_uuid,
                    QualityPredictionModelParameter.prediction_model_name == model_name,
                    QualityPredictionModelParameter.group == f"dist:{cluster_index}",
                )
            ).all()[0]
            == num_steps
        ):
            model_parameters = session.exec(
                select(QualityPredictionModelParameter.key).where(
                    QualityPredictionModelParameter.grid_uuid == grid_uuid,
                    QualityPredictionModelParameter.prediction_model_name == model_name,
                    QualityPredictionModelParameter.group == f"dist:{cluster_index}",
                )
            ).all()
        else:
            subquery = (
                select(
                    func.max(QualityPredictionModelParameter.timestamp).label("most_recent"),
                    QualityPredictionModelParameter.key,
                )
                .group_by(QualityPredictionModelParameter.key)
                .subquery("most_recent")
            )
            model_parameters = session.exec(
                select(QualityPredictionModelParameter).join(
                    subquery,
                    and_(
                        QualityPredictionModelParameter.grid_uuid == grid_uuid,
                        QualityPredictionModelParameter.prediction_model_name == model_name,
                        QualityPredictionModelParameter.group == f"dist:{cluster_index}",
                        QualityPredictionModelParameter.timestamp == subquery.c.most_recent,
                    ),
                )
            ).all()
    dist = np.zeros(num_steps)
    for mp in model_parameters:
        dist[int(mp.key)] = mp.value
    return dist


def _record_dist(dist: np.array, grid_uuid: str, cluster_index: int) -> None:
    if np.sum(dist) == 0:
        return None
    for j in range(len(dist)):
        publish_model_parameter_update(
            grid_uuid=grid_uuid,
            model_name=model_name,
            key=str(j),
            group=f"dist:{cluster_index}",
            value=dist[j],
        )
    return None


def _record_score(score: float, gridsquare_uuid: str) -> None:
    publish_gridsquare_model_prediction(
        gridsquare_uuid=gridsquare_uuid,
        model_name=model_name,
        prediction_value=score,
    )
    return None


def update(params: UpdateParameters) -> None:
    dist = _get_dist(params.grid_uuid, params.cluster_index)
    dist = update_distribution(dist, params.quality)
    _record_dist(dist, params.grid_uuid, params.cluster_index)
    post_update_score = score(dist, params.cluster_index)
    _record_score(post_update_score, params.gridsquare_uuid)
    return None
