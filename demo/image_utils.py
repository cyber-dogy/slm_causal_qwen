from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

from PIL import Image, ImageDraw, ImageFont, ImageOps


def ensure_rgb(image: Image.Image | None) -> Image.Image | None:
    if image is None:
        return None
    return image.convert("RGB")


def _get_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 20)
    except Exception:
        return ImageFont.load_default()


def build_labeled_tile(image: Image.Image, label: str, tile_size: Tuple[int, int] = (360, 280)) -> Image.Image:
    image = ImageOps.pad(ensure_rgb(image), tile_size, color=(245, 247, 250))
    font = _get_font()
    canvas = Image.new("RGB", (tile_size[0], tile_size[1] + 42), (250, 251, 253))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.size[0], 42), fill=(25, 42, 62))
    draw.text((16, 10), label, fill=(255, 255, 255), font=font)
    canvas.paste(image, (0, 42))
    return canvas


def build_montage(rgb_view1: Image.Image | None, rgb_view2: Image.Image | None, ir: Image.Image | None) -> Image.Image:
    tiles: List[Image.Image] = []
    for label, image in [
        ("RGB View 1", rgb_view1),
        ("RGB View 2", rgb_view2),
        ("IR After", ir),
    ]:
        if image is not None:
            tiles.append(build_labeled_tile(image, label))

    if not tiles:
        raise ValueError("At least one image is required to build a montage.")

    gap = 18
    width = sum(tile.size[0] for tile in tiles) + gap * (len(tiles) - 1)
    height = max(tile.size[1] for tile in tiles)
    canvas = Image.new("RGB", (width, height), (244, 246, 248))

    x = 0
    for tile in tiles:
        canvas.paste(tile, (x, 0))
        x += tile.size[0] + gap
    return canvas
