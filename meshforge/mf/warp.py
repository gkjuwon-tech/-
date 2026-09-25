"""Forward warping of a view with its own depth map into another turntable view (z-buffered)."""
import numpy as np

from .hull import C, SIZE


def warp(rgb, depth, mask, t_src, t_dst, off_src=0.0, off_dst=0.0, splat=1):
    v, u = np.where(mask)
    w = depth[v, u]
    rs, rd = np.deg2rad(t_src), np.deg2rad(t_dst)
    a = u - C - off_src
    X = a * np.cos(rs) + w * np.sin(rs)
    Z = -a * np.sin(rs) + w * np.cos(rs)
    ud = X * np.cos(rd) - Z * np.sin(rd) + C + off_dst
    wd = X * np.sin(rd) + Z * np.cos(rd)
    zbuf = np.full((SIZE, SIZE), -np.inf, np.float32)
    out = np.zeros((SIZE, SIZE, 3), np.uint8)
    for dy in range(splat):
        for dx in range(splat):
            ui = np.clip(np.floor(ud).astype(int) + dx, 0, SIZE - 1)
            vi = np.clip(v + dy, 0, SIZE - 1)
            np.maximum.at(zbuf, (vi, ui), wd)
    for dy in range(splat):
        for dx in range(splat):
            ui = np.clip(np.floor(ud).astype(int) + dx, 0, SIZE - 1)
            vi = np.clip(v + dy, 0, SIZE - 1)
            win = wd >= zbuf[vi, ui] - 1e-3
            out[vi[win], ui[win]] = rgb[v[win], u[win]]
    return out, np.isfinite(zbuf), zbuf
