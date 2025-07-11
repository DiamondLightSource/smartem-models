import typer
from backports.entry_points_selectable import entry_points

from smartem_models.utils.consumer_wrapper import consume


def start_service(level: str, service_name: str, queue_name: str):
    func = entry_points().select(group=f"smartem_models.{level}", name=service_name)[0].load()
    validation_model = entry_points().select(group=f"smartem_models.{level}.signature", name=service_name)[0].load()
    consume(func, queue_name, validation_model)


def run() -> None:
    typer.run(start_service)
    return None
