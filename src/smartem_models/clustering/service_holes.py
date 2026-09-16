import asyncio
import pickle
import uuid
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from smartem_backend import mq_publisher as mq_publisher_module
from smartem_backend.model.database import (
    CurrentQualityPrediction,
    FoilHole,
    FoilHoleGroup,
    FoilHoleGroupMembership,
    GridSquare,
    Micrograph,
    QualityMetric,
    QualityPredictionModelParameter,
)
from smartem_backend.mq_publisher import (
    publish_create_foilhole_group,
    publish_foilhole_group_model_prediction,
    publish_gridsquare_registered,
    publish_gridsquare_updated,
)
from smartem_backend.rmq import AioPikaPublisher
from smartem_backend.rmq.config import load_rmq_connection_url
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


def _cluster_group_uuid(grid_uuid: str, cluster_index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{grid_uuid}:{model_name}:cluster:{cluster_index}"))


def train_and_infer(
    params: InitParameters, diameter: float | None, square_imgs, foil_hole_positions, all_foil_holes, grid_uuid, engine
):
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

    if not params.init_scores_from_model:
        hist = [[0.5 for p in labelled_coords if p[2] == label] for label in range(num_clusters)]
    else:
        hist = []
        with Session(engine) as session:
            for label in range(num_clusters):
                other_model_scores_for_cluster = session.exec(
                    select(FoilHoleGroup, FoilHoleGroupMembership, CurrentQualityPrediction)
                    .where(FoilHoleGroup.grid_uuid == grid_uuid)
                    .where(FoilHoleGroup.name == str(label))
                    .where(FoilHoleGroupMembership.group_uuid == FoilHoleGroup.uuid)
                    .where(CurrentQualityPrediction.foilhole_uuid == FoilHoleGroupMembership.foilhole_uuid)
                    .where(CurrentQualityPrediction.prediction_model_name == params.init_scores_from_model)
                ).all()
            if len(other_model_scores_for_cluster) < 5:
                hist.append([0.5 for p in labelled_coords if p[2] == label])
            else:
                hist.append([q[2].value for q in other_model_scores_for_cluster])
    largest_cluster = np.max([len(p) for p in hist])
    hist = [np.pad(p, (0, largest_cluster - len(p)), "constant", constant_values=(np.nan, np.nan)) for p in hist]

    init_dists = init_distributions(hist)

    if params.model_output_dir:
        torch.save(model.state_dict(), params.model_output_dir / f"{grid_uuid}_{model_name}_model.pt")
        with open(params.model_output_dir / f"{grid_uuid}_{model_name}_kmeans.pkl", "wb") as pkl:
            pickle.dump(kmeans, pkl)

    return labelled_coords, init_dists


async def initialise(params: InitParameters) -> None:
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

    labelled_coords, init_dists = await asyncio.to_thread(
        train_and_infer, params, diameter, square_imgs, foil_hole_positions, all_foil_holes, grid_uuid, engine
    )

    # Create foil hole groups (one per cluster)
    cluster_holes: dict[int, list[str]] = {}
    for lc in labelled_coords:
        cluster_holes.setdefault(int(lc[2]), []).append(lc[0])

    publisher = AioPikaPublisher(
        url=load_rmq_connection_url(),
        exchange_name="smartem",
        routing_key="smartem",
    )
    await publisher.connect()
    mq_publisher_module.set_publisher(publisher)
    for cluster_idx, hole_uuids in cluster_holes.items():
        await publish_create_foilhole_group(
            grid_uuid=grid_uuid,
            foilhole_uuids=hole_uuids,
            group_uuid=_cluster_group_uuid(grid_uuid, cluster_idx),
        )

    await _set_model_parameters(
        init_dists,
        labelled_coords,
        grid_uuid,
        model=model_name,
    )

    post_update_scores = [_score(dist) for dist in init_dists]

    with Session(engine) as session:
        metric_names = [m.name for m in session.exec(select(QualityMetric)).all()]
    for cluster_idx, score_val in enumerate(post_update_scores):
        group_uuid = _cluster_group_uuid(grid_uuid, cluster_idx)
        for metric_name in metric_names:
            await publish_foilhole_group_model_prediction(
                group_uuid=group_uuid,
                model_name=model_name,
                prediction_value=score_val,
                metric=metric_name,
            )

    # after writing files used in inference check for grid squares that need to have inference run
    with Session(engine) as session:
        registered_grid_squares = session.exec(
            select(GridSquare)
            .where(GridSquare.grid_uuid == grid_uuid)
            .where(GridSquare.status == GridSquareStatus.REGISTERED)
        ).all()
    init_square_ids = [gs.uuid for gs in grid_squares]
    with Session(engine) as session:
        for rs in registered_grid_squares:
            if rs.uuid not in init_square_ids:
                try:
                    _ = await publish_gridsquare_registered(rs.uuid)
                except Exception as e:
                    print(e)
            else:
                rs.status = GridSquareStatus.FOIL_HOLES_DECISION_STARTED
                session.add(rs)
                _ = await publish_gridsquare_updated(
                    uuid=rs.uuid, grid_uuid=rs.grid_uuid, gridsquare_id=rs.gridsquare_id
                )
        session.commit()

    await publisher.close()

    return None


async def infer(params: InferenceParameters):
    if not params.model_output_dir.is_dir():
        return None
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
    model_path = params.model_output_dir / f"{grid_uuid}_{model_name}_model.pt"
    kmeans_path = params.model_output_dir / f"{grid_uuid}_{model_name}_kmeans.pkl"
    if not model_path.is_file() or not kmeans_path.is_file():
        return None
    torch.set_num_threads(params.num_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EIAE(
        alpha=torch.Tensor([[1.0, 1.0]]).to(device),
        input_dims=params.input_dim,
        hidden_dims=params.hidden_dims,
        lat_dim=params.latent_space_dim,
    )
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.to(device)
    model.eval()
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

    with open(kmeans_path, "rb") as pkl:
        kmeans = pickle.load(pkl)

    publisher = AioPikaPublisher(
        url=load_rmq_connection_url(),
        exchange_name="smartem",
        routing_key="smartem",
    )
    await publisher.connect()
    mq_publisher_module.set_publisher(publisher)

    kmeans.cluster_centers_ = kmeans.cluster_centers_.astype(np.float64)
    cluster_holes: dict[int, list[str]] = {}
    for huuid, coords in latent_coords.items():
        cluster_index = int(kmeans.predict(np.array([coords], dtype=np.float64))[0])
        await _add_cluster_index(grid_uuid, cluster_index, huuid, coords, model=model_name)
        cluster_holes.setdefault(cluster_index, []).append(huuid)

    with Session(engine) as session:
        metric_names = [m.name for m in session.exec(select(QualityMetric)).all()]

    for cluster_idx, hole_uuids in cluster_holes.items():
        await publish_create_foilhole_group(
            grid_uuid=grid_uuid,
            foilhole_uuids=hole_uuids,
            group_uuid=_cluster_group_uuid(grid_uuid, cluster_idx),
        )
        for metric_name in metric_names:
            dist = _get_dist(grid_uuid, cluster_idx, metric_name)
            await publish_foilhole_group_model_prediction(
                group_uuid=_cluster_group_uuid(grid_uuid, cluster_idx),
                model_name=model_name,
                prediction_value=_score(dist),
                metric=metric_name,
            )

    with Session(engine) as session:
        gs = session.exec(select(GridSquare).where(GridSquare.uuid == params.uuid)).one()
        gs.status = GridSquareStatus.FOIL_HOLES_DECISION_STARTED
        session.add(gs)
        session.commit()
        _ = await publish_gridsquare_updated(uuid=gs.uuid, grid_uuid=gs.grid_uuid, gridsquare_id=gs.gridsquare_id)

    await publisher.close()

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


async def update(params: UpdateParameters) -> None:
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

    publisher = AioPikaPublisher(
        url=load_rmq_connection_url(),
        exchange_name="smartem",
        routing_key="smartem",
    )
    await publisher.connect()
    mq_publisher_module.set_publisher(publisher)

    await publish_foilhole_group_model_prediction(
        group_uuid=_cluster_group_uuid(micrograph_chain[0].grid_uuid, cluster_index),
        model_name=model_name,
        prediction_value=_score(dist),
        metric=params.metric_name,
    )

    await publisher.close()

    return None
