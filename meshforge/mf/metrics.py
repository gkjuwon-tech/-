"""Ground-truth-free consistency checks for orthographic 0-elevation sheets.

M1: silhouette(theta) must equal mirror(silhouette(theta + 180)).
M2: the set of occupied rows must be identical in every view.
"""
import numpy as np


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


def row_extent(m):
    rows = np.where(m.any(axis=1))[0]
    return int(rows.min()), int(rows.max())


def best_hshift_iou(a, b, max_shift=80):
    """IoU of a vs b after the best horizontal shift of b (rotation axis unknown)."""
    best = (-1.0, 0)
    for s in range(-max_shift, max_shift + 1):
        v = iou(a, np.roll(b, s, axis=1))
        if v > best[0]:
            best = (v, s)
    return best


def m1_mirror(masks):
    out = {}
    for a, b in [(0, 180), (90, 270), (45, 225), (315, 135)]:
        if a in masks and b in masks:
            out[f"{a}|{b}"] = best_hshift_iou(masks[a], masks[b][:, ::-1])
    return out


def m2_rows(masks):
    ext = {az: row_extent(m) for az, m in masks.items()}
    tops = [e[0] for e in ext.values()]
    bots = [e[1] for e in ext.values()]
    height = np.median([b - t for t, b in ext.values()])
    spread = max(max(tops) - min(tops), max(bots) - min(bots))
    return ext, 1.0 - spread / height
