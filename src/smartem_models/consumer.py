import json
from functools import partial
from logging import getLogger

from backports.entry_points_selectable import entry_points
from pika.channel import Channel
from pika.frame import Body, Method
from pika.spec import BasicProperties
from pydantic import BaseModel
from smartem_backend.model.mq_event import MessageQueueEventType
from smartem_backend.utils import setup_rabbitmq

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
    return message["count"]


def on_message(
    channel: Channel,
    method: Method,
    properties: BasicProperties,
    body: Body,
    hooks: dict | None = None,
    signatures: dict | None = None,
    config: dict | None = None,
):
    hooks = hooks or {}
    signatures = signatures or {}
    config = config or {}
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
    success = True

    try:
        event = MessageQueueEventType(event_type)
    except ValueError:
        logger.warning(f"Event type {event_type} not recognised", exc_info=True)
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        return
    for hook in hooks.get(event_type):
        ParameterModel = signatures.get(event_type, {}).get(hook.name).load()
        params = ParameterModel(**message, **config.get("model_parameters", {}).get(hook.name, {}).get(event_type, {}))
        if config.get("distributed", {}).get(hook.name, {}).get(event_type):
            requested_counts: list[int] | None
            if (requested_counts := config.get("act_on_count", {}).get(hook.name, {}).get(event_type)) is not None:
                if count_functions[event_type](message) not in requested_counts:
                    continue
            if (minimum_count := config.get("minimum_count", {}).get(hook.name, {}).get(event_type)) is not None:
                if count_functions[event_type](message) < minimum_count:
                    continue
            try:
                publish_request(
                    config.get("processing_queues", {}).get(hook.name, {}).get(event_type, ""),
                    event,
                    params,
                )
            except ValueError:
                success = False
                continue
        else:
            hook.load()(params)

    if success:
        channel.basic_ack(delivery_tag=method.delivery_tag)
    else:
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


def run():
    pub, con = setup_rabbitmq(queue_name="smartem_models")
    config = get_config()
    eps = {
        k.replace("smartem_models.", ""): v
        for k, v in entry_points().items()
        if k.startswith("smartem_models") and not k.endswith("signature")
    }
    signatures = {
        k.replace("smartem_models.", "").replace(".signature", ""): v
        for k, v in entry_points().items()
        if k.startswith("smartem_models") and k.endswith("signature")
    }
    unwrapped_signatures = {}
    for k, v in signatures.items():
        unwrapped_signatures[k] = {w.name: w for w in v}
    con.consume(partial(on_message, hooks=eps, signatures=unwrapped_signatures, config=config), prefetch_count=1)
