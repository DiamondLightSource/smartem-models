from pathlib import Path

import numpy as np
import torch
from pydantic import BaseModel
from smartem_decisions.model.database import GridSquare, QualityPredictionModelParameter
from smartem_decisions.utils import setup_postgres_connection
from sqlmodel import Session, and_, func, select
from torch.utils.data import DataLoader
from torchvision import transforms

from smartem_models.clustering.calldata import SquareDataset, prepare_image
from smartem_models.clustering.grid_clustering import train
from smartem_models.clustering.models import EIAE
from smartem_models.utils import read_img
from smartem_models.utils.parameter_updating_cluster import update_distributions

model_name = "vae-square"


class InitParameters(BaseModel):
    grid_id: int
    batch_size: int = 16
    seed: int = 10
    input_model: Path | None = None
    input_dim: tuple[int, ...] = (1, 64, 64)
    hidden_dims: tuple[int, ...] = (1, 16, 32, 64, 128)
    latent_space_dim: int = 2
    learning_rate: float = 0.0005
    num_epochs: int = 1000
    model_output_path: str = ""


def initialise(params: InitParameters) -> None:
    engine = setup_postgres_connection()
    with Session(engine) as session:
        grid_squares = session.exec(select(GridSquare).where(GridSquare.grid_id == params.grid_id)).all()
    if not all(gs.gridsquare_img for gs in grid_squares):
        return None
    square_imgs = {gs.id: Path(gs.gridsquare_img) for gs in grid_squares}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_x = SquareDataset(square_imgs, transform=transforms.Resize(params.input_dim[-1], antialias=True))
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
        model.load_state_dict(torch.load(params.input_model, map_location=torch.device("cpu")))
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
        coords = model(sample["x1"])[0].detach().cpu().numpy()
        for i, label in enumerate(sample["label"].detach().cpu().numpy().flatten()):
            latent_coords[label] = coords[i]

    if params.model_output_path:
        torch.save(model.state_dict(), params.model_output_path)

    return None


class InferenceParameters(BaseModel):
    model_path: Path
    gridsquare_img_path: Path
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
    return coords


class UpdateParameters(BaseModel):
    quality: bool
    cluster_index: int
    grid_id: int


def _get_dist(grid_id: int, cluster_index: int, num_steps: int = 10) -> np.array:
    engine = setup_postgres_connection()
    with Session(engine) as session:
        # need to deal with the timestamps here !!!!
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
                    QualityPredictionModelParameter.grid_id == 1,
                    QualityPredictionModelParameter.prediction_model_name == "test",
                    QualityPredictionModelParameter.timestamp == subquery.c.most_recent,
                ),
            )
        ).all()
    dist = np.zeros(num_steps)
    for mp in model_parameters:
        dist[int(mp.key)] = mp.value
    return dist


def _record_dist(dist: np.array, grid_id: int, cluster_index: int, num_steps: int = 10) -> None:
    engine = setup_postgres_connection()
    with Session(engine) as session:
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
                    QualityPredictionModelParameter.grid_id == 1,
                    QualityPredictionModelParameter.prediction_model_name == "test",
                    QualityPredictionModelParameter.timestamp == subquery.c.most_recent,
                ),
            )
        ).all()
        for mp in model_parameters:
            mp.value = dist[int(mp.key)]
        session.add_all(model_parameters)
        session.commit()
    return None


def update(params: UpdateParameters):
    dist = _get_dist(params.grid_id, params.cluster_index)
    dist = update_distributions(dist, params.cluster_index, params.quality)
    _record_dist(dist, params.grid_id, params.cluster_index)
