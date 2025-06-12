import json
from logging import getLogger
from pathlib import Path

from backports.entry_points_selectable import entry_points
from pika.channel import Channel
from pika.frame import Body, Method
from pika.spec import BasicProperties
from pydantic import BaseModel, model_validator
from smartem_decisions.utils import setup_rabbitmq

from smartem_models.utils import get_config

logger = getLogger("smartem_models.consumer")


class TrainingParameters(BaseModel):
    grid_id: int


class InferenceParameters(BaseModel):
    img_path: Path
    magnification_scale: str
    model_weights_path: Path | None = None


class UpdateParameters(BaseModel):
    quality: bool
    gridsquare_id: int | None = None
    foilhole_id: int | None = None

    @model_validator(model="after")
    def id_present_check(self):
        if sum((self.gridsquare_id is None, self.foilhole_id is None)) != 1:
            raise ValueError("Either gridsquare_id or foilhole_id must be provided but not both")
        return self


def on_message(channel: Channel, method: Method, properties: BasicProperties, body: Body):
    message = json.loads(body.decode())
    if "event_type" not in message:
        logger.warning(f"Message missing 'event_type' field: {message}")
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        return

    event_type = message["event_type"]
    registered_models = get_config().get("registered_models", [])

    match event_type:
        case "train_model":
            training_hooks = [
                e for e in entry_points().select(group="smartem_models.train") if e.name in registered_models
            ]
            for hook in training_hooks:
                hook.load()(TrainingParameters(grid_id=message["grid_id"]))
        case "infer":
            inference_hooks = [
                e for e in entry_points().select(group="smartem_models.infer") if e.name in registered_models
            ]
            for hook in inference_hooks:
                hook.load()(
                    InferenceParameters(
                        img_path=message["img_path"],
                        magnification_scale=message["magnification_scale"],
                        model_weights_path=message.get("model_weights_path"),
                    )
                )
        case "update":
            update_hooks = [e for e in entry_points().select("smartem_models.update") if e.name in registered_models]
            for hook in update_hooks:
                hook.load()(
                    UpdateParameters(
                        quality=message["quality"],
                        gridsquare_id=message.get("gridsquare_id"),
                        foilhole_id=message.get("foilhole_id"),
                    )
                )


def run():
    pub, con = setup_rabbitmq(queue_name="smartem_models")
    con.consume(on_message, prefetch_count=1)
