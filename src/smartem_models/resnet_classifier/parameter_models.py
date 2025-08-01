from pathlib import Path

from pydantic import BaseModel


class InferenceParameters(BaseModel):
    grid_uuid: str
    model_path: Path
    cpus: int = 4
