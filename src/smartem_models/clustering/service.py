from pathlib import Path

import numpy as np
import torch
from pydantic import BaseModel
from smartem_decisions.model.database import GridSquare
from smartem_decisions.utils import setup_postgres_connection
from sqlmodel import Session, select
from torch.utils.data import DataLoader
from torchvision import transforms

from smartem_models.clustering.calldata import SquareDataset
from smartem_models.clustering.grid_clustering import train
from smartem_models.clustering.models import EIAE


class InitParameters(BaseModel):
    grid_id: int
    batch_size: int = 16
    seed: int = 10
    input_model: Path | None = None
    input_dim: tuple[int] = (1, 64, 64)
    hidden_dims: tuple[int] = (1, 16, 32, 64, 128)
    latent_space_dim: int = 2
    learning_rate: float = 0.0005
    num_epochs: int = 1000


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
    epoch_losses = [10**15]

    for epoch in range(params.num_epochs):
        loss_record, epoch_loss = train(epoch, train_dataloader, model, optimizer, device)
        losses += loss_record
        epoch_losses.append(epoch_loss)

    return None
