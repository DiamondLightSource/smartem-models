from pathlib import Path

from pydantic import BaseModel


class InitParameters(BaseModel):
    grid_uuid: str
    batch_size: int = 16
    seed: int = 10
    input_model: Path | None = None
    input_dim: tuple[int, ...] = (1, 32, 32)
    hidden_dims: tuple[int, ...] = (1, 16, 32, 64, 128)
    latent_space_dim: int = 2
    learning_rate: float = 0.0005
    num_epochs: int = 500
    model_output_path: str = ""
    kmeans_output_path: str = ""
    num_threads: int = 2


class InferenceParameters(BaseModel):
    uuid: str  # gridsquare uuid
    model_path: Path
    kmeans_path: str
    input_dim: tuple[int, ...] = (1, 64, 64)
    hidden_dims: tuple[int, ...] = (1, 16, 32, 64, 128)
    latent_space_dim: int = 2
    num_threads: int = 2


class UpdateParameters(BaseModel):
    quality: bool
    cluster_index: int
    grid_uuid: str
    foilhole_uuid: str
