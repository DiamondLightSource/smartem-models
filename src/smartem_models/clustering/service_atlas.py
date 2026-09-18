import asyncio
import pickle
import uuid
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from smartem_backend import mq_publisher as mq_publisher_module
from smartem_backend.model.database import (
    Atlas,
    FoilHole,
    Grid,
    GridSquare,
    Micrograph,
    QualityMetric,
    QualityPredictionModelParameter,
)
from smartem_backend.mq_publisher import (
    publish_atlas_model_prediction,
    publish_create_gridsquare_group,
    publish_gridsquare_group_model_prediction,
)
from smartem_backend.rmq import AioPikaPublisher
from smartem_backend.rmq.config import load_rmq_connection_url
from smartem_backend.utils import setup_postgres_connection
from sqlmodel import Session, and_, func, select
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from smartem_models.clustering.calldata import SquareAtlasMagDataset
from smartem_models.clustering.grid_clustering import train
from smartem_models.clustering.models import EIAE
from smartem_models.clustering.parameter_models_atlas import InitParameters, UpdateParameters
from smartem_models.clustering.service import _set_model_parameters
from smartem_models.utils.parameter_updating_cluster import init_distributions, update_distribution_from_prob

model_name = "dae-atlas"


def _cluster_group_uuid(grid_uuid: str, cluster_index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{grid_uuid}:{model_name}:cluster:{cluster_index}"))


def _find_atlas_image(parent: Path) -> Path:
    if mrcs := list(parent.glob("Atlas_*.mrc")):
        return mrcs[0]
    if tiffs := list(parent.glob("Atlas_*.tiff")):
        return tiffs[0]
    if mrcs := list(parent.glob("*_atlas.mrc")):
        return mrcs[0]
    if tiffs := list(parent.glob("*_atlas.tiff")):
        return tiffs[0]
    raise FileNotFoundError(f"No atlas image found in {parent}")


def _score(dist):
    step = 1 / len(dist)
    midpoints = np.arange(step, 1 + step, step)
    return np.sum(step * midpoints * dist)


def train_and_infer(params: InitParameters, grid_squares, atlas_path, square_width):
    torch.set_num_threads(params.num_threads)
    square_positions = {gs.uuid: (gs.center_x, gs.center_y) for gs in grid_squares if gs.center_x is not None}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_x = SquareAtlasMagDataset(
        Path(atlas_path),
        square_positions,
        square_width,
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
    params.num_epochs = 150
    for _epoch in range(params.num_epochs):
        loss_record, epoch_loss = train(train_dataloader, model, optimizer, device)
        losses += loss_record
        epoch_losses.append(epoch_loss)

    model.eval()

    latent_coords = {}
    for sample in train_dataloader:
        coords = np.nan_to_num(model.encode(Variable(sample["x1"]).to(device))[0].detach().cpu().numpy())
        for i, label in enumerate(sample["lab"].detach().cpu().numpy().flatten()):
            latent_coords[grid_squares[label].uuid] = coords[i]

    num_clusters = len(latent_coords) // 5
    if num_clusters > 10:
        num_clusters = 10
    if not num_clusters:
        return None
    labelled_coords = list(latent_coords.items())
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(
        np.array([p[1] for p in labelled_coords])
    )
    labelled_coords = [(p[0], p[1], q) for p, q in zip(labelled_coords, kmeans.labels_, strict=False)]
    hist = [[0.5 for p in labelled_coords if p[2] == label] for label in range(num_clusters)]
    largest_cluster = np.max([len(p) for p in hist])
    hist = [np.pad(p, (0, largest_cluster - len(p)), "constant", constant_values=(np.nan, np.nan)) for p in hist]
    init_dists = init_distributions(hist)

    if params.model_output_dir:
        torch.save(model.state_dict(), params.model_output_dir / f"{params.grid_uuid}_{model_name}_model.pt")
        with open(params.model_output_dir / f"{params.grid_uuid}_{model_name}_kmeans.pkl", "wb") as pkl:
            pickle.dump(kmeans, pkl)

    return labelled_coords, init_dists


async def initialise(params: InitParameters) -> None:
    engine = setup_postgres_connection()
    with Session(engine) as session:
        atlas = session.exec(select(Atlas).where(Atlas.grid_uuid == params.grid_uuid)).all()[-1]
        grid_squares = session.exec(select(GridSquare).where(GridSquare.grid_uuid == params.grid_uuid)).all()
        grid = session.exec(select(Grid).where(Grid.uuid == params.grid_uuid)).all()[0]
        if Path(grid.atlas_dir).is_dir():
            atlas_path = Path(grid.atlas_dir) / f"{grid.name}_montage.mrc"
        else:
            atlas_path = _find_atlas_image(Path(grid.atlas_dir).parent)
        s = int(
            1.1
            * np.max([np.max([gs.size_width for gs in grid_squares]), np.max([gs.size_height for gs in grid_squares])])
        )

    labelled_coords, init_dists = await asyncio.to_thread(train_and_infer, params, grid_squares, atlas_path, s)

    # Create gridsquare groups (one per cluster)
    cluster_gridsquares: dict[int, list[str]] = {}
    for lc in labelled_coords:
        cluster_gridsquares.setdefault(int(lc[2]), []).append(lc[0])

    publisher = AioPikaPublisher(
        url=load_rmq_connection_url(),
        exchange_name="smartem",
        routing_key="smartem",
    )
    await publisher.connect()
    mq_publisher_module.set_publisher(publisher)
    for cluster_idx, gridsquare_uuids in cluster_gridsquares.items():
        await publish_create_gridsquare_group(
            grid_uuid=params.grid_uuid,
            gridsquare_uuids=gridsquare_uuids,
            group_uuid=_cluster_group_uuid(params.grid_uuid, cluster_idx),
        )

    await _set_model_parameters(init_dists, labelled_coords, params.grid_uuid, model=model_name)

    post_update_scores = [_score(dist) for dist in init_dists]

    with Session(engine) as session:
        metric_names = [m.name for m in session.exec(select(QualityMetric)).all()]
    for cluster_idx, score_val in enumerate(post_update_scores):
        group_uuid = _cluster_group_uuid(params.grid_uuid, cluster_idx)
        for metric_name in metric_names:
            await publish_gridsquare_group_model_prediction(
                group_uuid=group_uuid,
                model_name=model_name,
                prediction_value=score_val,
                metric=metric_name,
            )

    await publish_atlas_model_prediction(atlas.uuid, -1, model_name=model_name)
    await publisher.close()

    return None


def _get_dist(grid_uuid: str, cluster_index: int, metric_name: str | None = None, num_steps: int = 10) -> np.array:
    engine = setup_postgres_connection()
    keys = [str(k) for k in range(num_steps)]
    common_filters = (
        QualityPredictionModelParameter.grid_uuid == grid_uuid,
        QualityPredictionModelParameter.prediction_model_name == model_name,
        QualityPredictionModelParameter.group == f"dist:{int(cluster_index)}",
        QualityPredictionModelParameter.metric_name == metric_name,  # noqa: E711
        QualityPredictionModelParameter.key.in_(keys),
    )
    with Session(engine) as session:
        latest_per_key = (
            select(
                func.max(QualityPredictionModelParameter.timestamp).label("most_recent"),
                QualityPredictionModelParameter.key,
            )
            .where(*common_filters)
            .group_by(QualityPredictionModelParameter.key)
            .subquery()
        )
        model_parameters = session.exec(
            select(QualityPredictionModelParameter)
            .join(
                latest_per_key,
                and_(
                    QualityPredictionModelParameter.key == latest_per_key.c.key,
                    QualityPredictionModelParameter.timestamp == latest_per_key.c.most_recent,
                ),
            )
            .where(*common_filters)
        ).all()
    dist = np.zeros(num_steps)
    for mp in model_parameters:
        dist[int(mp.key)] = mp.value
    return dist


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
            .where(QualityPredictionModelParameter.key == micrograph_chain[0].uuid)
            .where(QualityPredictionModelParameter.prediction_model_name == model_name)
            .order_by(QualityPredictionModelParameter.timestamp.desc())
        ).all()
        if not cluster_index_response:
            return None
        cluster_index = int(cluster_index_response[0].value)
        grid_uuid = micrograph_chain[0].grid_uuid

    dist = _get_dist(grid_uuid, cluster_index, metric_name=params.metric_name)
    dist = update_distribution_from_prob(dist, params.quality)

    with Session(engine) as session:
        for j in range(len(dist)):
            session.add(
                QualityPredictionModelParameter(
                    grid_uuid=grid_uuid,
                    prediction_model_name=model_name,
                    key=str(j),
                    group=f"dist:{cluster_index}",
                    value=dist[j],
                    metric_name=params.metric_name,
                )
            )
        session.commit()

    post_update_score = _score(dist)

    publisher = AioPikaPublisher(
        url=load_rmq_connection_url(),
        exchange_name="smartem",
        routing_key="smartem",
    )
    await publisher.connect()
    mq_publisher_module.set_publisher(publisher)

    await publish_gridsquare_group_model_prediction(
        group_uuid=_cluster_group_uuid(grid_uuid, cluster_index),
        model_name=model_name,
        prediction_value=post_update_score,
        metric=params.metric_name,
    )

    await publisher.close()

    return None
