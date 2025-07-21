import pickle
from pathlib import Path

import numpy as np
import torch
from pydantic import BaseModel
from sklearn.cluster import KMeans
from smartem_decisions.model.database import FoilHole, GridSquare, QualityPrediction, QualityPredictionModelParameter
from smartem_decisions.utils import setup_postgres_connection
from sqlmodel import Session, select
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from smartem_models.clustering.calldata import HoleDataset, prepare_image
from smartem_models.clustering.grid_clustering import train
from smartem_models.clustering.models import EIAE
from smartem_models.utils import read_img
from smartem_models.utils.parameter_updating_cluster import init_distributions, score, update_distribution

model_name = "vae-hole"


def _set_model_parameters(
    dists: np.array,
    coords: list[tuple[str, tuple[float, float], int]],
    grid_uuid: str,
    model_weights_path: str,
    kmeans_path: str,
) -> None:
    model_parameters = []
    for i, d in enumerate(dists):
        model_parameters.extend(
            [
                QualityPredictionModelParameter(
                    grid_uuid=grid_uuid,
                    prediction_model_name=model_name,
                    key=str(j),
                    group=f"dist:{i}",
                    value=float(d[j]),
                )
                for j in range(len(d))
            ]
        )
    coord_parameters = []
    cluster_parameters = []
    for c in coords:
        coord_parameters.append(
            QualityPredictionModelParameter(
                grid_uuid=grid_uuid,
                prediction_model_name=model_name,
                key="x",
                group=f"coordinates:{c[0]}",
                value=float(c[1][0]),
            )
        )
        coord_parameters.append(
            QualityPredictionModelParameter(
                grid_uuid=grid_uuid,
                prediction_model_name=model_name,
                key="y",
                group=f"coordinates:{c[0]}",
                value=float(c[1][1]),
            )
        )
        cluster_parameters.append(
            QualityPredictionModelParameter(
                grid_uuid=grid_uuid,
                prediction_model_name=model_name,
                key=str(c[0]),
                group="cluster_indices",
                value=float(c[2]),
            )
        )
    engine = setup_postgres_connection()
    with Session(engine) as session:
        session.add_all(model_parameters)
        session.add_all(coord_parameters)
        session.add_all(cluster_parameters)
        session.commit()
    return None


class InitParameters(BaseModel):
    grid_uuid: str
    batch_size: int = 16
    seed: int = 10
    input_model: Path | None = None
    input_dim: tuple[int, ...] = (1, 32, 32)
    hidden_dims: tuple[int, ...] = (1, 16, 32, 64, 128)
    latent_space_dim: int = 2
    learning_rate: float = 0.0005
    num_epochs: int = 1000
    model_output_path: str = ""
    kmeans_output_path: str = ""


def initialise(params: InitParameters) -> None:
    engine = setup_postgres_connection()
    with Session(engine) as session:
        grid_squares = session.exec(select(GridSquare).where(GridSquare.grid_uuid == params.grid_uuid)).all()
    grid_squares = [gs for gs in grid_squares if gs.image_path]
    if not all(gs.image_path for gs in grid_squares):
        return None

    square_imgs = []
    foil_hole_positions = []
    all_foil_holes = []
    diameter: int | None = None
    with Session(engine) as session:
        for gs in grid_squares:
            square_imgs.append(Path(gs.image_path))
            foil_holes = session.exec(select(FoilHole).where(FoilHole.gridsquare_uuid == gs.uuid)).all()
            foil_holes = [fh for fh in foil_holes if not fh.is_near_grid_bar]
            all_foil_holes.extend(foil_holes)
            if diameter is None:
                diameter = foil_holes[0].diameter
            foil_hole_positions.append([(fh.x_location, fh.y_location) for fh in foil_holes])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if diameter is None:
        return None

    train_x = HoleDataset(
        square_imgs,
        foil_hole_positions,
        int(1.1 * diameter),
        transform=transforms.Resize(params.input_dim[-1], antialias=True),
    )
    train_dataloader = DataLoader(train_x, batch_size=params.batch_size, shuffle=True, pin_memory=True)

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

    evaluate_dataloader = DataLoader(train_x, batch_size=1, shuffle=False, pin_memory=True)
    latent_coords = {}
    for i, sample in enumerate(evaluate_dataloader):
        coords = model.encode(Variable(sample["x1"]).to(device))[0].detach().cpu().numpy()
        latent_coords[all_foil_holes[i].uuid] = coords[0]

    num_clusters = 10 if len(latent_coords) > 50 else len(latent_coords) // 5
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

    _set_model_parameters(
        init_dists, labelled_coords, params.grid_uuid, params.model_output_path, params.kmeans_output_path
    )

    return None


def _add_cluster_index(grid_uuid: str, cluster_index: int, gridsquare: str, coords: np.array) -> None:
    coords_parameters = [
        QualityPredictionModelParameter(
            grid_uuid=grid_uuid,
            prediction_model_name=model_name,
            key="x",
            group=f"coordinates:{gridsquare}",
            value=coords[0],
        ),
        QualityPredictionModelParameter(
            grid_uuid=grid_uuid,
            prediction_model_name=model_name,
            key="y",
            group=f"coordinates:{gridsquare}",
            value=coords[1],
        ),
    ]
    cluster_parameter = QualityPredictionModelParameter(
        grid_uuid=grid_uuid,
        prediction_model_name=model_name,
        key=gridsquare,
        group="cluster_indices",
        value=cluster_index,
    )
    engine = setup_postgres_connection()
    with Session(engine) as session:
        session.add_all(coords_parameters)
        session.add(cluster_parameter)
        session.commit()
    return None


class InferenceParameters(BaseModel):
    grid_uuid: str
    model_path: Path
    gridsquare_img_path: Path
    gridsquare_uuid: str
    kmeans_path: str
    input_dim: tuple[int, ...] = (1, 64, 64)
    hidden_dims: tuple[int, ...] = (1, 16, 32, 64, 128)
    latent_space_dim: int = 2


def infer(params: InferenceParameters):
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
        model_parameters = []
        for key in range(num_steps):
            model_parameters.append(
                session.exec(
                    select(QualityPredictionModelParameter)
                    .where(QualityPredictionModelParameter.grid_uuid == grid_uuid)
                    .where(QualityPredictionModelParameter.prediction_model_name == model_name)
                    .where(QualityPredictionModelParameter.group == f"dist:{cluster_index}")
                    .where(QualityPredictionModelParameter.key == str(key))
                    .order_by(QualityPredictionModelParameter.timestamp.desc())
                ).first()
            )
    dist = np.zeros(num_steps)
    for mp in model_parameters:
        dist[int(mp.key)] = mp.value
    return dist


def _record_dist(dist: np.array, grid_uuid: str, cluster_index: int) -> None:
    model_parameters = [
        QualityPredictionModelParameter(
            grid_uuid=grid_uuid,
            prediction_model_name=model_name,
            key=str(j),
            group=f"dist:{cluster_index}",
            value=dist[j],
        )
        for j in range(len(dist))
    ]
    engine = setup_postgres_connection()
    with Session(engine) as session:
        session.add_all(model_parameters)
        session.commit()
    return None


def _record_score(score: float, foilhole_uuid: str, grid_uuid: str, cluster_index: int) -> None:
    prediction = QualityPrediction(foilhole_uuid=foilhole_uuid, prediction_model_name=model_name, value=score)
    engine = setup_postgres_connection()
    with Session(engine) as session:
        square_uuids = session.exec(
            select(QualityPredictionModelParameter)
            .where(QualityPredictionModelParameter.prediction_model_name == model_name)
            .where(QualityPredictionModelParameter.grid_uuid == grid_uuid)
            .where(QualityPredictionModelParameter.group == "cluster_indices")
            .where(QualityPredictionModelParameter.value == cluster_index)
        ).all()
        synched_predictions = [
            QualityPrediction(foilhole_uuid=s.key, prediction_model_name=model_name, value=score) for s in square_uuids
        ]
        session.add(prediction)
        session.add_all(synched_predictions)
        session.commit()
    return None


class UpdateParameters(BaseModel):
    quality: bool
    cluster_index: int
    grid_uuid: str
    foilhole_uuid: str


def update(params: UpdateParameters) -> None:
    dist = _get_dist(params.grid_uuid, params.cluster_index)
    dist = update_distribution(dist, params.quality)
    _record_dist(dist, params.grid_uuid, params.cluster_index)
    post_update_score = score(dist, params.cluster_index)
    _record_score(post_update_score, params.foilhole_uuid, params.grid_uuid, params.cluster_index)
    return None
