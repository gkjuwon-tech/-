"""CPU renders of the voxel proxy: shaded 'clay' guides and previews from any direction."""
import numpy as np
from scipy import ndimage

from .hull import SIZE, C


def rot(azimuth, elevation=0.0):
    """Rows of the returned matrix are the camera (right, down, toward-camera) axes in world
    coords, consistent with hull.project for elevation 0 (u = X cos t - Z sin t, v = Y)."""
    t, e = np.deg2rad(azimuth), np.deg2rad(elevation)
    right = np.array([np.cos(t), 0.0, -np.sin(t)])
    fwd = np.array([np.sin(t), 0.0, np.cos(t)])          # toward camera at elevation 0
    down = np.array([0.0, 1.0, 0.0])
    # tilt camera up by `elevation`: toward-camera vector rotates toward -Y (world up is -Y)
    toward = np.cos(e) * fwd - np.sin(e) * down
    down2 = np.cos(e) * down + np.sin(e) * fwd
    return np.stack([right, down2, toward])


def clay(grid, occ, azimuth, elevation=0.0, off=0.0, cy=C):
    """Lambert-shaded depth render of the occupied voxels (light from the camera, upper-left)."""
    P = np.stack([grid.X[occ], grid.Y[occ] - cy, grid.Z[occ]], -1)
    R = rot(azimuth, elevation)
    q = P @ R.T
    u = np.clip(np.round(q[:, 0] + C + off).astype(int), 0, SIZE - 1)
    v = np.clip(np.round(q[:, 1] + cy).astype(int), 0, SIZE - 1)
    depth = np.full((SIZE, SIZE), -np.inf, np.float32)
    np.maximum.at(depth, (v, u), q[:, 2])
    depth = ndimage.grey_dilation(depth, size=(grid.step + 1, grid.step + 1))
    sil = np.isfinite(depth)
    d = np.where(sil, depth, np.nan)
    d = ndimage.gaussian_filter(np.nan_to_num(d, nan=np.nanmin(d)), 1.2)
    gy, gx = np.gradient(d)
    n = np.stack([gx, gy, np.ones_like(d)], -1)   # toward-camera normal (image coords)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    light = np.array([-0.4, -0.5, 0.77]); light /= np.linalg.norm(light)
    shade = np.clip((n @ light), 0, 1) * 0.8 + 0.2
    img = np.full((SIZE, SIZE, 3), 128, np.uint8)
    img[sil] = (np.clip(shade[sil], 0, 1)[:, None] * [235, 228, 215]).astype(np.uint8)
    return img, sil
