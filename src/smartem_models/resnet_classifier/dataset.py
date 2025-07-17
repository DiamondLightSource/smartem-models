from functools import lru_cache
from pathlib import Path

import mrcfile
import numpy as np
import tifffile
from PIL import Image
from pydantic import BaseModel, ConfigDict
from smartem_decisions.model.database import Atlas, AtlasTile, AtlasTileGridSquarePosition, GridSquare
from smartem_decisions.utils import setup_postgres_connection
from sqlmodel import Session, select


@lru_cache(maxsize=50)
def _get_tile_image(base_filename: str, file_format: str, atlas_dir: str) -> Image.Image:
    tile_file_name = f"{base_filename}.{file_format.lower()}"
    tile_file = Path(atlas_dir) / tile_file_name
    if tile_file.suffix == ".mrc":
        data = mrcfile.read(tile_file)
    else:
        data = tifffile.imread(tile_file)
    mean = np.mean(data)
    sdev = np.std(data)
    sigma_min = mean - 3 * sdev
    sigma_max = mean + 3 * sdev
    data = np.ndarray.copy(data)
    data[data < sigma_min] = sigma_min
    data[data > sigma_max] = sigma_max
    data = data - data.min()
    data = data / data.max()
    data = data * 255
    array = data.astype("uint8")
    tile_img = Image.fromarray(array)
    return tile_img


class GridSquarePosition(BaseModel):
    image: Image.Image
    center_on_atlas: tuple[int, int]

    model_config = ConfigDict(arbitrary_types_allowed=True)


def grid_square_positions(grid_uuid: str, atlas_dir: str) -> dict[str, list[GridSquarePosition]]:
    engine = setup_postgres_connection()
    with Session(engine) as session:
        positions = session.exec(
            select(Atlas, AtlasTile, AtlasTileGridSquarePosition, GridSquare)
            .where(Atlas.grid_uuid == grid_uuid)
            .where(Atlas.uuid == AtlasTile.atlas_uuid)
            .where(AtlasTile.uuid == AtlasTileGridSquarePosition.atlastile_uuid)
            .where(AtlasTileGridSquarePosition.gridsquare_uuid == GridSquare.uuid)
        ).all()

    gs_imgs: dict[str, list[Image.Image]] = {}

    for pos in positions:
        d = (
            pos[2].center_x - pos[2].size_width // 2,
            pos[2].center_y - pos[2].size_height // 2,
            pos[2].center_x + pos[2].size_width // 2,
            pos[2].center_y + pos[2].size_height // 2,
        )
        if gs_imgs.get(pos[3].uuid) is None:
            gs_imgs[pos[3].uuid] = [
                GridSquarePosition(
                    images=_get_tile_image(pos[1].base_filename, pos[1].file_format, atlas_dir).crop(d),
                    centers=(pos[3].center_x, pos[3].center_y),
                )
            ]
        else:
            gs_imgs[pos[3].uuid].append(
                GridSquarePosition(
                    images=_get_tile_image(pos[1].base_filename, pos[1].file_format, atlas_dir).crop(d),
                    centers=(pos[3].center_x, pos[3].center_y),
                )
            )

    return gs_imgs
