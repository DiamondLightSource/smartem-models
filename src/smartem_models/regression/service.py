from pathlib import Path

import torch
from pydantic import BaseModel
from smartem_decisions.model.database import FoilHole, GridSquare
from smartem_decisions.utils import setup_postgres_connection
from sqlmodel import Session, select
from torch.utils.data import DataLoader
from torchvision import transforms

from smartem_models.regression.dataset import FoilHoleDataset
from smartem_models.regression.model import ResNetRegression


class InitParameters(BaseModel):
    grid_uuid: str
    model_weights_path: Path
    batch_size: int = 16


def initialise(params: InitParameters) -> None:
    engine = setup_postgres_connection()
    with Session(engine) as session:
        grid_squares = session.exec(select(GridSquare).where(GridSquare.grid_uuid == params.grid_uuid)).all()
    if not all(gs.gridsquare_img for gs in grid_squares):
        return None
    square_imgs = {}
    foil_hole_positions = {}
    with Session(engine) as session:
        for gs in grid_squares:
            square_imgs[gs.uuid] = Path(gs.image_path)
            foil_holes = session.exec(select(FoilHole).where(FoilHole.gridsquare_uuid == gs.uuid)).all()
            foil_hole_positions[gs.uuid] = [(fh.x_location, fh.y_location, fh.diameter) for fh in foil_holes]

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = FoilHoleDataset(square_imgs, foil_hole_positions, transform=transform)
    loader = DataLoader(dataset, batch_size=params.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetRegression(freeze_base=True).to(device)

    model.load_state_dict(torch.load(params.model_weights_path, map_location=device))
    model.eval()

    predictions = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            predictions.extend(outputs.cpu().numpy().flatten())

    return None
