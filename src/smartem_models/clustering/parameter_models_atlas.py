from pathlib import Path

from pydantic import BaseModel


class InitParameters(BaseModel):
    grid_uuid: str
    batch_size: int = 64
    seed: int = 10
    input_model: Path | None = None
    input_dim: tuple[int, ...] = (1, 64, 64)
    hidden_dims: tuple[int, ...] = (1, 16, 32, 64, 128)
    latent_space_dim: int = 2
    learning_rate: float = 0.0005
    num_epochs: int = 250
    model_output_dir: Path | None = None
    num_threads: int = 2


class UpdateParameters(BaseModel):
    quality: float
    micrograph_uuid: str
    metric_name: str | None = None
