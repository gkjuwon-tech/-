"""Turnaround sheet layout: 2x3 grid, orthographic, 0 deg elevation."""
import numpy as np
from PIL import Image

AZIMUTHS = [0, 45, 90, 180, 270, 315]  # row-major cell order
BG_RGB = (128, 128, 128)


def split_cells(img, rows=2, cols=3):
    """Return {azimuth: HxWx3 uint8 array} for each grid cell."""
    a = np.asarray(img.convert("RGB"))
    h, w = a.shape[0] // rows, a.shape[1] // cols
    cells = {}
    for i, az in enumerate(AZIMUTHS):
        r, c = divmod(i, cols)
        cells[az] = a[r * h:(r + 1) * h, c * w:(c + 1) * w]
    return cells


def load_cells(path):
    return split_cells(Image.open(path))
