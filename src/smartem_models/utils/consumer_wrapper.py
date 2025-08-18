import json
from collections.abc import Callable
from threading import Thread

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

            def _func_in_thread():
                func(message)
                con._connection.add_callback_threadsafe(lambda: channel.basic_ack(method.delivery_tag))

            t = Thread(target=_func_in_thread)
            t.start()
        except Exception:
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            return

    con.consume(on_message, prefetch_count=1)
