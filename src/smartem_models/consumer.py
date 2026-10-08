import asyncio
import inspect
import json
from functools import partial
from logging import getLogger

from aio_pika.abc import AbstractIncomingMessage
from backports.entry_points_selectable import entry_points
from pydantic import BaseModel
from smartem_backend import mq_publisher as mq_publisher_module
from smartem_backend.model.mq_event import MessageQueueEventType
from smartem_backend.rmq import AioPikaConsumer, AioPikaPublisher
from smartem_backend.rmq.config import load_rmq_connection_url
from smartem_backend.utils import setup_postgres_connection

from smartem_models.utils import get_config

logger = getLogger("smartem_models.consumer")

engine = setup_postgres_connection()

# one publisher per processing queue, opened on first use and closed when the consumer stops
_request_publishers: dict[str, AioPikaPublisher] = {}


async def _get_request_publisher(queue_name: str) -> AioPikaPublisher:
    if (publisher := _request_publishers.get(queue_name)) is None:
        publisher = AioPikaPublisher(url=load_rmq_connection_url(), exchange_name="", routing_key=queue_name)
        try:
            await publisher.connect()
        except Exception:
            await publisher.close()
            raise
        _request_publishers[queue_name] = publisher
    return publisher


async def _close_request_publishers() -> None:
    while _request_publishers:
        _, publisher = _request_publishers.popitem()
        try:
            await publisher.close()
        except Exception:
            logger.warning("Failed to close request publisher", exc_info=True)


async def publish_request(
    queue_name: str,
    message_type: MessageQueueEventType,
    message_body: BaseModel,
):
    if not queue_name:
        logger.error("No queue name was provided to processing request publish", exc_info=True)
        raise ValueError("No queue name provided")
    publisher = await _get_request_publisher(queue_name)
    await publisher.publish_event(message_type, message_body)


def _gridsquare_create_count(message: dict) -> int:
    return message["count"]


async def on_message(
    message: AbstractIncomingMessage,
    hooks: dict | None = None,
    signatures: dict | None = None,
    config: dict | None = None,
):
    hooks = hooks or {}
    signatures = signatures or {}
    config = config or {}
    body = json.loads(message.body.decode())
    if "event_type" not in body:
        logger.warning(f"Message missing 'event_type' field: {body}")
        await message.nack(requeue=False)
        return

    count_functions = {
        "gridsquare.created": _gridsquare_create_count,
        "gridsquare.registered": _gridsquare_create_count,
    }

    event_type = body["event_type"]

    try:
        event = MessageQueueEventType(event_type)
    except ValueError:
        logger.warning(f"Event type {event_type} not recognised", exc_info=True)
        await message.nack(requeue=False)
        return

    for hook in hooks.get(event_type, []):
        ParameterModel = signatures.get(event_type, {}).get(hook.name).load()
        params = ParameterModel(**body, **config.get("model_parameters", {}).get(hook.name, {}).get(event_type, {}))
        if config.get("distributed", {}).get(hook.name, {}).get(event_type):
            requested_counts: list[int] | None
            if (requested_counts := config.get("act_on_count", {}).get(hook.name, {}).get(event_type)) is not None:
                if count_functions[event_type](body) not in requested_counts:
                    continue
            if (minimum_count := config.get("minimum_count", {}).get(hook.name, {}).get(event_type)) is not None:
                if count_functions[event_type](body) < minimum_count:
                    continue
            try:
                await publish_request(
                    config.get("processing_queues", {}).get(hook.name, {}).get(event_type, ""),
                    event,
                    params,
                )
            except ValueError:
                logger.warning(f"Failed to publish request for {hook.name}")
                # channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
                # return
                continue
        else:
            result = hook.load()(params)
            if inspect.isawaitable(result):
                await result

    await message.ack()


async def async_run():
    url = load_rmq_connection_url()

    con = AioPikaConsumer(url=url, queue_name="smartem_models", exchange_name="", prefetch_count=1)
    # hooks that are run in this process rather than distributed publish their results through this
    publisher = AioPikaPublisher(
        url=url,
        exchange_name="smartem",
        routing_key="smartem",
        exchange_type="fanout",
    )
    config = get_config()
    eps = {
        k.replace("smartem_models.", ""): v
        for k, v in entry_points().items()
        if k.startswith("smartem_models") and not k.endswith("signature")
    }
    registered_models = config.get("registered_models", [])
    for k, v in eps.items():
        eps[k] = [w for w in v if w.name in registered_models]
    signatures = {
        k.replace("smartem_models.", "").replace(".signature", ""): v
        for k, v in entry_points().items()
        if k.startswith("smartem_models") and k.endswith("signature")
    }
    unwrapped_signatures = {}
    for k, v in signatures.items():
        unwrapped_signatures[k] = {w.name: w for w in v}
    try:
        await publisher.connect()
        mq_publisher_module.set_publisher(publisher)
        await con.connect()
        await con.consume(partial(on_message, hooks=eps, signatures=unwrapped_signatures, config=config))
    finally:
        mq_publisher_module.set_publisher(None)
        await _close_request_publishers()
        await publisher.close()
        await con.close()


def run():
    asyncio.run(async_run())
