import json
from logging import getLogger

from backports.entry_points_selectable import entry_points
from pika.channel import Channel
from pika.frame import Body, Method
from pika.spec import BasicProperties
from pydantic import BaseModel
from smartem_backend.model.database import GridSquare
from smartem_backend.model.mq_event import MessageQueueEventType
from smartem_backend.utils import setup_postgres_connection, setup_rabbitmq
from sqlalchemy import func
from sqlmodel import Session, select

from smartem_models.utils import get_config

logger = getLogger("smartem_models.consumer")


def publish_request(
    queue_name: str,
    message_type: MessageQueueEventType,
    message_body: BaseModel,
):
    if not queue_name:
        logger.error("No queue name was provided to processing request publish", exc_info=True)
        raise ValueError("No queue name provided")
    pub, con = setup_rabbitmq(queue_name=queue_name)
    pub.publish_event(message_type, message_body)


def _gridsquare_create_count(message: dict) -> int:
    engine = setup_postgres_connection()
    with Session(engine) as session:
        grid_uuid = session.exec(select(GridSquare).where(GridSquare.uuid == message["uuid"])).one().grid_uuid
        num_gridsquares = session.exec(
            select(func.count(GridSquare.uuid))
            .where(GridSquare.grid_uuid == grid_uuid)
            .where(GridSquare.image_path.is_not(None))
        ).one()
    return num_gridsquares


def on_message(channel: Channel, method: Method, properties: BasicProperties, body: Body):
    message = json.loads(body.decode())
    if "event_type" not in message:
        logger.warning(f"Message missing 'event_type' field: {message}")
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        return

    count_functions = {
        "gridsquare.created": _gridsquare_create_count,
        "gridsquare.registered": _gridsquare_create_count,
    }

    event_type = message["event_type"]
    config = get_config()
    registered_models = config.get("registered_models", [])

    try:
        event = MessageQueueEventType(event_type)
    except ValueError:
        logger.warning(f"Event type {event_type} not recognised", exc_info=True)
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        return
    training_hooks = [
        e for e in entry_points().select(group=f"smartem_models.{event_type}") if e.name in registered_models
    ]
    for hook in training_hooks:
        ParameterModel = entry_points().select(group=f"smartem_models.{event_type}.signature", name=hook.name)[0].load()
        params = ParameterModel(**message, **config.get("model_parameters", {}).get(hook.name, {}).get(event_type, ""))
        if config.get("distributed", {}).get(hook.name, {}).get(event_type):
            requested_counts: list[int] | None
            if (requested_counts := config.get("act_on_count", {}).get(hook.name, {}).get(event_type)) is not None:
                if count_functions[event_type](message) not in requested_counts:
                    break
            if (minimum_count := config.get("minimum_count", {}).get(hook.name, {}).get(event_type)) is not None:
                if count_functions[event_type](message) < minimum_count:
                    break
            try:
                publish_request(
                    config.get("processing_queues", {}).get(hook.name, {}).get(event_type, ""),
                    event,
                    params,
                )
            except ValueError:
                channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
                return
        else:
            hook.load()(params)

    channel.basic_ack(delivery_tag=method.delivery_tag)


def run():
    pub, con = setup_rabbitmq(queue_name="smartem_models")
    con.consume(on_message, prefetch_count=1)
