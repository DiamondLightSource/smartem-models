from pathlib import Path

from pydantic import BaseModel


class InferenceParameters(BaseModel):
    grid_uuid: str
    model_path: Path
    cpus: int = 4


class HoleInferenceParameters(BaseModel):
    uuid: str  # this is the gridsquare uuid
    model_path: Path
    cpus: int = 4
    update_latent_reps: bool = True
