"""Foreground masks against the flat gray background."""
import numpy as np
from scipy import ndimage

from .sheet import BG_RGB


def mask_from_bg(cell, tol=18):
    diff = np.abs(cell.astype(int) - np.array(BG_RGB)).max(axis=2)
    m = diff > tol
    m = ndimage.binary_opening(m, iterations=1)
    m = ndimage.binary_fill_holes(m)
    lab, n = ndimage.label(m)
    if n > 1:  # keep largest component
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = lab == (1 + int(np.argmax(sizes)))
    return m


def canonicalize(mask, cell=None, top=32, bottom=480, size=512):
    """Similarity-normalize a view: in an orthographic 0-elevation sheet every view
    shares the same top and bottom row, so map [top, bottom] of the mask onto fixed
    rows and center the silhouette horizontally. Returns (mask, cell) resampled."""
    from PIL import Image

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    t, b = rows.min(), rows.max()
    s = (bottom - top) / max(b - t, 1)
    cx = (cols.min() + cols.max()) / 2
    # affine: out(x, y) samples in(x', y') with x' = (x - size/2)/s + cx, y' = (y - top)/s + t
    coeffs = (1 / s, 0, cx - (size / 2) / s, 0, 1 / s, t - top / s)
    m = Image.fromarray(mask.astype(np.uint8) * 255).transform(
        (size, size), Image.AFFINE, coeffs, resample=Image.BILINEAR)
    out_mask = np.asarray(m) > 127
    out_cell = None
    if cell is not None:
        out_cell = np.asarray(Image.fromarray(cell).transform(
            (size, size), Image.AFFINE, coeffs, resample=Image.BICUBIC, fillcolor=BG_RGB))
    return out_mask, out_cell
