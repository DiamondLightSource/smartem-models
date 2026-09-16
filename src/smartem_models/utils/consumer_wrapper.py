from collections.abc import Callable

from pydantic import BaseModel
from smartem_backend.rmq import AioPikaConsumer, decode_event_body
from smartem_backend.rmq.config import load_rmq_connection_url, load_rmq_topology


async def consume(func: Callable, queue_name: str, message_format: type[BaseModel]):
    url = load_rmq_connection_url()
    exchange_name, _queue_name = load_rmq_topology()

    con = AioPikaConsumer(url=url, queue_name=queue_name, exchange_name="", prefetch_count=1)
    await con.connect()

    async def on_message(message):
        message_body = decode_event_body(message)
        try:
            await func(message_format(**message_body))
            await message.ack()
        except Exception:
            await message.reject(requeue=False)

    await con.consume(on_message)
