from pathlib import Path

import numpy as np
import torch
from pydantic import BaseModel
from smartem_decisions.model.database import Grid
from smartem_decisions.utils import setup_postgres_connection
from sqlmodel import Session, select
from torchvision import models, transforms

from smartem_models.resnet_classifier.dataset import GridSquarePosition, grid_square_positions
from smartem_models.resnet_classifier.model import Net


class InferenceParameters(BaseModel):
    grid_uuid: str
    model_path: Path
    cpus: int = 4


def infer(params: InferenceParameters) -> None:
    torch.set_num_threads(params.cpus)
    feature_extractor = models.resnet18(pretrained=True)
    feature_extractor.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    feature_extractor.maxpool = torch.nn.Identity()
    feature_extractor.fc = torch.nn.Identity()
    model = Net(
        feature_extractor,
        num_classes=2,
        centroid_size=512,
        model_output_size=512,
        length_scale=0.1,
        gamma=0.999,
    )
    model.load_state_dict(torch.load(params.model_path, map_location=torch.device("cpu")))

    engine = setup_postgres_connection()
    with Session(engine) as session:
        grid = session.exec(select(Grid).where(Grid.uuid == params.grid_uuid)).one()
    gs_positions = grid_square_positions(params.grid_uuid, str(Path(grid.atlas_dir).parent))

    boundaries = (
        (
            np.min([pos[0].center_on_atlas[0] for pos in gs_positions.values()]),
            np.max([pos[0].center_on_atlas[0] for pos in gs_positions.values()]),
        ),
        (
            np.min([pos[0].center_on_atlas[1] for pos in gs_positions.values()]),
            np.max([pos[0].center_on_atlas[1] for pos in gs_positions.values()]),
        ),
    )

    def _boundary_check(gs_name: str) -> tuple[bool, bool, bool, bool]:
        loc = gs_positions[gs_name][0].center_on_atlas
        brange = (
            boundaries[0][1] - boundaries[0][0],
            boundaries[1][1] - boundaries[1][0],
        )
        thresholds = (
            (
                boundaries[0][0] + 0.075 * brange[0],
                boundaries[0][1] - 0.075 * brange[0],
            ),
            (
                boundaries[1][0] + 0.075 * brange[1],
                boundaries[1][1] - 0.075 * brange[1],
            ),
        )
        return (
            loc[0] > thresholds[0][0]
            and loc[0] < thresholds[0][1]
            and loc[1] > thresholds[1][0]
            and loc[1] < thresholds[1][1]
        )

    def _area(positions: list[GridSquarePosition]) -> int:
        return np.sum([(pos.image.size[0] * pos.image.size[1]) for pos in positions])

    if np.std([_area(tmps) for tmps in gs_positions.values()]) > 50:
        size_threshold = np.quantile(np.array([_area(tmps) for tmps in gs_positions.values()]), 0.5)
    else:
        size_threshold = 0

    scores = {}
    for s, pos in gs_positions.items():
        img_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        score: float = 0
        total_size_x = 0
        total_size_y = 0
        images = [p.image for p in pos]
        if _area(images) < size_threshold:
            scores[s] = 1
            continue
        for template in images:
            inputs = np.array(template)
            inputs = inputs.astype("int16")
            inputs = inputs - inputs.min()
            inputs = (inputs * (255.0 / inputs.max())).astype(np.uint8)
            tmp = torch.Tensor(inputs)
            tmp = torch.reshape(tmp, (1, tmp.shape[0], tmp.shape[1]))
            tmp = tmp.repeat(3, 1, 1)
            tmp = img_transform(tmp)
            tmp = (tmp - tmp.min()) / tmp.max()
            img_variable = tmp.unsqueeze(0)

            resnet_img, y_pred = model(img_variable)
            confidence = y_pred.max(1)[0].detach().cpu().numpy()
            _, predicted = torch.max(y_pred.data, 1)
            predicted = predicted.detach().cpu().numpy()
            predicted = -2.0 * predicted + 1.0
            score += (predicted * confidence)[0]
            total_size_x += template.size[0]
            total_size_y += template.size[1]

        score /= len(pos)
        scores[s] = (score * np.sqrt(total_size_x * total_size_y) * (1 if _boundary_check(s) else 0.5),)

    return None
