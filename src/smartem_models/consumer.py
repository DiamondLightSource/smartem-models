import json
from logging import getLogger
from pathlib import Path

from backports.entry_points_selectable import entry_points
from pika.channel import Channel
from pika.frame import Body, Method
from pika.spec import BasicProperties
from pydantic import BaseModel, model_validator
from smartem_decisions.model.mq_event import MessageQueueEventType
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


def publish_request(
    queue_name: str,
    message_type: MessageQueueEventType,
    message_body: TrainingParameters | InferenceParameters | UpdateParameters,
):
    if not queue_name:
        logger.error("No queue name was provided to processing request publish", exc_info=True)
        raise ValueError("No queue name provided")
    pub, con = setup_rabbitmq(queue_name=queue_name)
    pub.publish_event(message_type, message_body)


def on_message(channel: Channel, method: Method, properties: BasicProperties, body: Body):
    message = json.loads(body.decode())
    if "event_type" not in message:
        logger.warning(f"Message missing 'event_type' field: {message}")
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        return

    event_type = message["event_type"]
    config = get_config()
    registered_models = config.get("registered_models", [])

    match event_type:
        case "train_model":
            training_hooks = [
                e for e in entry_points().select(group="smartem_models.train") if e.name in registered_models
            ]
            for hook in training_hooks:
                params = TrainingParameters(grid_id=message["grid_id"])
                if config.get("distributed", {}).get("train", {}).get(hook.name):
                    try:
                        publish_request(
                            config.get("processing_queues", {}).get(hook.name, {}).get("train", ""),
                            MessageQueueEventType.TRAIN,
                            params,
                        )
                    except ValueError:
                        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
                else:
                    hook.load()(params)
        case "infer":
            inference_hooks = [
                e for e in entry_points().select(group="smartem_models.infer") if e.name in registered_models
            ]
            for hook in inference_hooks:
                params = InferenceParameters(
                    img_path=message["img_path"],
                    magnification_scale=message["magnification_scale"],
                    model_weights_path=message.get("model_weights_path"),
                )
                if config.get("distributed").get("infer", {}).get(hook.name):
                    try:
                        publish_request(
                            config.get("processing_queues", {}).get(hook.name, {}).get("infer", ""),
                            MessageQueueEventType.INFER,
                            params,
                        )
                    except ValueError:
                        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
                else:
                    hook.load()(params)
        case "update":
            update_hooks = [e for e in entry_points().select("smartem_models.update") if e.name in registered_models]
            for hook in update_hooks:
                params = UpdateParameters(
                    quality=message["quality"],
                    gridsquare_id=message.get("gridsquare_id"),
                    foilhole_id=message.get("foilhole_id"),
                )
                if config.get("distributed").get("update", {}).get(hook.name):
                    try:
                        publish_request(
                            config.get("processing_queues", {}).get(hook.name, {}).get("update", ""),
                            MessageQueueEventType.MODEL_UPDATE,
                            params,
                        )
                    except ValueError:
                        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
                else:
                    hook.load()(params)
        case _:
            logger.warning(f"Event type {event_type} not recognised", exc_info=True)
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    channel.basic_ack(delivery_tag=method.delivery_tag)


def run():
    pub, con = setup_rabbitmq(queue_name="smartem_decisions")
    con.consume(on_message, prefetch_count=1)
