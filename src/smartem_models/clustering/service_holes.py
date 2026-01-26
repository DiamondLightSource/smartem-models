import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from smartem_backend.model.database import (
    FoilHole,
    GridSquare,
    Micrograph,
    QualityMetric,
    QualityPredictionModelParameter,
)
from smartem_backend.mq_publisher import publish_gridsquare_registered, publish_multi_foilhole_model_prediction
from smartem_backend.utils import setup_postgres_connection
from smartem_common.entity_status import GridSquareStatus
from sqlmodel import Session, select
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from smartem_models.clustering.calldata import HoleDataset
from smartem_models.clustering.grid_clustering import train
from smartem_models.clustering.models import EIAE
from smartem_models.clustering.parameter_models_holes import InferenceParameters, InitParameters, UpdateParameters
from smartem_models.clustering.service import _add_cluster_index, _set_model_parameters
from smartem_models.utils.parameter_updating_cluster import init_distributions, update_distribution_from_prob

model_name = "dae-hole"


def initialise(params: InitParameters) -> None:
    engine = setup_postgres_connection()
    with Session(engine) as session:
        grid_uuid = session.exec(select(GridSquare).where(GridSquare.uuid == params.uuid)).one().grid_uuid
        grid_squares = session.exec(select(GridSquare).where(GridSquare.grid_uuid == grid_uuid)).all()
    grid_squares = [gs for gs in grid_squares if gs.image_path]
    grid_squares = grid_squares[: params.num_squares]
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
            if diameter is None and foil_holes:
                diameter = foil_holes[0].diameter
            foil_hole_positions.append(
                [(fh.x_location, fh.y_location) for fh in foil_holes if fh.x_location is not None]
            )

    torch.set_num_threads(params.num_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if diameter is None:
        return None

    train_x = HoleDataset(
        square_imgs,
        foil_hole_positions,
        int(1.1 * diameter),
        transform=transforms.Resize(params.input_dim[-1], antialias=True),
    )
    train_dataloader = DataLoader(train_x, batch_size=params.batch_size, shuffle=True, pin_memory=True, drop_last=True)

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

    train_x = HoleDataset(
        square_imgs,
        foil_hole_positions,
        int(1.1 * diameter),
        transform=transforms.Resize(params.input_dim[-1], antialias=True),
        subset=False,
    )
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

    # after writing files used in inference check for grid squares that need to have inference run
    with Session(engine) as session:
        registered_grid_squares = session.exec(
            select(GridSquare)
            .where(GridSquare.grid_uuid == grid_uuid)
            .where(GridSquare.status == GridSquareStatus.REGISTERED)
        ).all()
    init_square_ids = [gs.uuid for gs in grid_squares]
    for rs in registered_grid_squares:
        if rs.uuid not in init_square_ids:
            try:
                _ = publish_gridsquare_registered(rs.uuid)
            except Exception as e:
                print(e)

    _set_model_parameters(
        init_dists,
        labelled_coords,
        grid_uuid,
        model=model_name,
    )

    post_update_scores = [_score(dist) for dist in init_dists]

    with Session(engine) as session:
        metric_names = [m.name for m in session.exec(select(QualityMetric)).all()]
    for lc in labelled_coords:
        for metric_name in metric_names:
            _record_score(post_update_scores[lc[2]], [lc[0]], metric_name=metric_name, model=model_name)

    return None


def infer(params: InferenceParameters):
    if not params.model_path.is_file() or not params.kmeans_path.is_file():
        return None
    torch.set_num_threads(params.num_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EIAE(
        alpha=torch.Tensor([[1.0, 1.0]]).to(device),
        input_dims=params.input_dim,
        hidden_dims=params.hidden_dims,
        lat_dim=params.latent_space_dim,
    )
    model.load_state_dict(torch.load(params.model_path, weights_only=True, map_location=device))
    model.to(device)
    model.eval()
    engine = setup_postgres_connection()
    foil_hole_positions = []
    all_foil_holes = []
    diameter: int | None = None
    with Session(engine) as session:
        gs = session.exec(select(GridSquare).where(GridSquare.uuid == params.uuid)).one()
        grid_uuid = gs.grid_uuid
        gridsquare_img_path = Path(gs.image_path)
        foil_holes = session.exec(select(FoilHole).where(FoilHole.gridsquare_uuid == gs.uuid)).all()
        foil_holes = [fh for fh in foil_holes if not fh.is_near_grid_bar]
        all_foil_holes.extend(foil_holes)
        if diameter is None:
            diameter = foil_holes[0].diameter
        foil_hole_positions.append([(fh.x_location, fh.y_location) for fh in foil_holes if fh.x_location is not None])
    if diameter is None:
        return None
    train_x = HoleDataset(
        [gridsquare_img_path],
        foil_hole_positions,
        int(1.1 * diameter),
        transform=transforms.Resize(params.input_dim[-1], antialias=True),
        subset=False,
    )
    evaluate_dataloader = DataLoader(train_x, batch_size=1, shuffle=False, pin_memory=True)
    latent_coords = {}
    for i, sample in enumerate(evaluate_dataloader):
        coords = model.encode(Variable(sample["x1"]).to(device))[0].detach().cpu().numpy()
        latent_coords[all_foil_holes[i].uuid] = coords[0]

    with open(params.kmeans_path, "rb") as pkl:
        kmeans = pickle.load(pkl)
    kmeans.cluster_centers_ = kmeans.cluster_centers_.astype(np.float64)
    for huuid, coords in latent_coords.items():
        cluster_index = kmeans.predict(np.array([coords], dtype=np.float64))
        _add_cluster_index(grid_uuid, cluster_index, huuid, coords, model=model_name)

    with Session(engine) as session:
        metric_names = [m.name for m in session.exec(select(QualityMetric)).all()]
    for fhuuid in latent_coords.keys():
        for metric_name in metric_names:
            dist = _get_dist(grid_uuid, cluster_index, metric_name)
            _record_score(_score(dist), [fhuuid], metric_name=metric_name, model=model_name)

    return coords


def _get_dist(grid_uuid: str, cluster_index: int, metric_name: str | None = None, num_steps: int = 10) -> np.array:
    engine = setup_postgres_connection()
    with Session(engine) as session:
        model_parameters = []
        for key in range(num_steps):
            model_parameters.append(
                session.exec(
                    select(QualityPredictionModelParameter)
                    .where(QualityPredictionModelParameter.grid_uuid == grid_uuid)
                    .where(QualityPredictionModelParameter.prediction_model_name == model_name)
                    .where(QualityPredictionModelParameter.group == f"dist:{int(cluster_index)}")
                    .where(QualityPredictionModelParameter.metric_name == metric_name)
                    .where(QualityPredictionModelParameter.key == str(key))
                    .order_by(QualityPredictionModelParameter.timestamp.desc())
                ).first()
            )
    dist = np.zeros(num_steps)
    for mp in model_parameters:
        if mp is not None:
            dist[int(mp.key)] = mp.value
    return dist


def _score(dist):
    step = 1 / len(dist)
    midpoints = np.arange(step, 1 + step, step)
    return np.sum(step * midpoints * dist)


def _record_score(
    score: float, foilhole_uuids: list[str], metric_name: str | None = None, model: str = model_name
) -> None:
    publish_multi_foilhole_model_prediction(
        foilhole_uuids=foilhole_uuids,
        model_name=model,
        prediction_value=score,
        metric=metric_name,
    )
    return None


def update(params: UpdateParameters) -> None:
    engine = setup_postgres_connection()
    with Session(engine) as session:
        micrograph_chain = session.exec(
            select(GridSquare, FoilHole, Micrograph)
            .where(Micrograph.uuid == params.micrograph_uuid)
            .where(FoilHole.uuid == Micrograph.foilhole_uuid)
            .where(GridSquare.uuid == FoilHole.gridsquare_uuid)
        ).one()
        cluster_index_response = session.exec(
            select(QualityPredictionModelParameter)
            .where(QualityPredictionModelParameter.grid_uuid == micrograph_chain[0].grid_uuid)
            .where(QualityPredictionModelParameter.group == "cluster_indices")
            .where(QualityPredictionModelParameter.key == micrograph_chain[1].uuid)
            .where(QualityPredictionModelParameter.prediction_model_name == model_name)
            .order_by(QualityPredictionModelParameter.timestamp.desc())
        ).all()
        if not cluster_index_response:
            return None
        cluster_index = int(cluster_index_response[0].value)
        foil_holes = {
            p.key
            for p in session.exec(
                select(QualityPredictionModelParameter)
                .where(QualityPredictionModelParameter.grid_uuid == micrograph_chain[0].grid_uuid)
                .where(QualityPredictionModelParameter.group == "cluster_indices")
                .where(QualityPredictionModelParameter.value == cluster_index)
                .where(QualityPredictionModelParameter.prediction_model_name == model_name)
                .order_by(QualityPredictionModelParameter.timestamp.desc())
            ).all()
        }

    dist = _get_dist(micrograph_chain[0].grid_uuid, cluster_index, metric_name=params.metric_name)
    dist = update_distribution_from_prob(dist, params.quality)

    with Session(engine) as session:
        for j in range(len(dist)):
            session.add(
                QualityPredictionModelParameter(
                    grid_uuid=micrograph_chain[0].grid_uuid,
                    prediction_model_name=model_name,
                    key=str(j),
                    group=f"dist:{cluster_index}",
                    value=dist[j],
                    metric_name=params.metric_name,
                )
            )
        session.commit()

    post_update_score = _score(dist)
    _record_score(post_update_score, list(foil_holes), metric_name=params.metric_name, model=model_name)
    return None
