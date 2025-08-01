import json
from collections.abc import Callable

from pika.channel import Channel
from pika.frame import Body, Method
from pika.spec import BasicProperties
from pydantic import BaseModel
from smartem_backend.utils import setup_rabbitmq


def consume(func: Callable, queue_name: str, message_format: type[BaseModel]):
    pub, con = setup_rabbitmq(queue_name=queue_name)

    def on_message(channel: Channel, method: Method, properties: BasicProperties, body: Body):
        message = message_format(**json.loads(body.decode()))
        try:
            func(message)
        except Exception:
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        channel.basic_ack(delivery_tag=method.delivery_tag)

    con.consume(on_message, prefetch_count=1)
