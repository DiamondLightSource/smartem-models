import numpy as np
import torch
from smartem_backend.model.database import FoilHole, GridSquare
from smartem_backend.mq_publisher import publish_foilhole_model_prediction
from smartem_backend.utils import setup_postgres_connection
from sqlmodel import Session, select
from torchvision import models, transforms

from smartem_models.resnet_classifier.model import Net
from smartem_models.resnet_classifier.parameter_models import HoleInferenceParameters
from smartem_models.utils import read_img

model_name = "resnet-holes"


def infer(params: HoleInferenceParameters) -> None:
    torch.set_num_threads(params.cpus)
    feature_extractor = models.resnet18(pretrained=False)
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
    model.eval()

    engine = setup_postgres_connection()

    with Session(engine) as session:
        grid_square = session.exec(select(GridSquare).where(GridSquare.uuid == params.gridsquare_uuid)).all()[0]
        if not grid_square.image_path:
            return None
        gs_img = read_img(grid_square.image_path)
        foil_holes = session.exec(select(FoilHole).where(FoilHole.gridsquare_uuid == params.gridsquare_uuid)).all()
        foil_holes = [fh for fh in foil_holes if not fh.is_near_grid_bar]
        diameter = foil_holes[0].diameter
        foil_hole_positions = {fh.uuid: (fh.x_location, fh.y_location) for fh in foil_holes}

    img_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    scores = {}
    for h, pos in foil_hole_positions.items():
        score: float = 0
        template = gs_img[
            pos[0] - int(1.1 * diameter // 2) : pos[0] + int(1.1 * diameter // 2),
            pos[1] - int(1.1 * diameter // 2) : pos[1] + int(1.1 * diameter // 2),
        ]
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
        predicted = 2.0 * predicted - 1
        score += 0.5 * ((predicted * confidence)[0] + 1)

        score /= len(pos)
        scores[h] = score

    max_score = np.max(list(scores.values()))

    for k, v in scores.items():
        publish_foilhole_model_prediction(gridsquare_uuid=k, model_name=model_name, prediction_value=v / max_score)

    return None
