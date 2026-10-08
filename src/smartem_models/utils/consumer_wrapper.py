from collections.abc import Callable

from pydantic import BaseModel
from smartem_backend import mq_publisher as mq_publisher_module
from smartem_backend.rmq import AioPikaConsumer, AioPikaPublisher, decode_event_body
from smartem_backend.rmq.config import load_rmq_connection_url


async def consume(func: Callable, queue_name: str, message_format: type[BaseModel]):
    url = load_rmq_connection_url()

    con = AioPikaConsumer(url=url, queue_name=queue_name, exchange_name="", prefetch_count=1)
    # a single publisher is shared by all handlers for the lifetime of the process
    publisher = AioPikaPublisher(
        url=url,
        exchange_name="smartem",
        routing_key="smartem",
        exchange_type="fanout",
    )

    async def on_message(message):
        message_body = decode_event_body(message)
        try:
            await func(message_format(**message_body))
            await message.ack()
        except Exception:
            await message.reject(requeue=False)

    try:
        await publisher.connect()
        mq_publisher_module.set_publisher(publisher)
        await con.connect()
        await con.consume(on_message)
    finally:
        mq_publisher_module.set_publisher(None)
        await publisher.close()
        await con.close()
