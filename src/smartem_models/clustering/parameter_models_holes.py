from pathlib import Path

from pydantic import BaseModel


class InitParameters(BaseModel):
    uuid: str
    batch_size: int = 16
    seed: int = 10
    input_model: Path | None = None
    input_dim: tuple[int, ...] = (1, 32, 32)
    hidden_dims: tuple[int, ...] = (1, 16, 32, 64, 128)
    latent_space_dim: int = 2
    learning_rate: float = 0.0005
    num_epochs: int = 500
    model_output_dir: Path | None = None
    num_threads: int = 2
    num_squares: int | None = 5
    init_scores_from_model: str | None = None


class InferenceParameters(BaseModel):
    uuid: str  # gridsquare uuid
    model_output_dir: Path
    input_dim: tuple[int, ...] = (1, 32, 32)
    hidden_dims: tuple[int, ...] = (1, 16, 32, 64, 128)
    latent_space_dim: int = 2
    num_threads: int = 2


class UpdateParameters(BaseModel):
    quality: float
    micrograph_uuid: str
    metric_name: str | None = None
