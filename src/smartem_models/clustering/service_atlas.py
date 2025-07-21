import pickle
from pathlib import Path

import numpy as np
import torch
from pydantic import BaseModel
from sklearn.cluster import KMeans
from smartem_decisions.model.database import Acquisition, Grid, GridSquare
from smartem_decisions.utils import setup_postgres_connection
from sqlmodel import Session, select
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from smartem_models.clustering.calldata import SquareAtlasMagDataset
from smartem_models.clustering.grid_clustering import train
from smartem_models.clustering.models import EIAE
from smartem_models.clustering.service import _set_model_parameters
from smartem_models.utils.parameter_updating_cluster import init_distributions

model_name = "dae-atlas"


class InitParameters(BaseModel):
    grid_uuid: str
    batch_size: int = 16
    seed: int = 10
    input_model: Path | None = None
    input_dim: tuple[int, ...] = (1, 64, 64)
    hidden_dims: tuple[int, ...] = (1, 16, 32, 64, 128)
    latent_space_dim: int = 2
    learning_rate: float = 0.0005
    num_epochs: int = 500
    model_output_path: str = ""
    kmeans_output_path: str = ""


def initialise(params: InitParameters) -> None:
    engine = setup_postgres_connection()
    with Session(engine) as session:
        grid_squares = session.exec(select(GridSquare).where(GridSquare.grid_uuid == params.grid_uuid)).all()
        atlas_path = (
            session.exec(
                select(Grid, Acquisition)
                .where(Grid.uuid == params.grid_uuid)
                .where(Grid.acquisition_uuid == Acquisition.uuid)
            )
            .all()[0][1]
            .atlas_path
        )
        s = int(
            1.1
            * np.max([np.max([gs.size_width for gs in grid_squares]), np.max([gs.size_height for gs in grid_squares])])
        )

    square_positions = {gs.uuid: (gs.center_x, gs.center_y) for gs in grid_squares if gs.center_x is not None}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_x = SquareAtlasMagDataset(
        Path(atlas_path), square_positions, s, transform=transforms.Resize(params.input_dim[-1], antialias=True)
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
