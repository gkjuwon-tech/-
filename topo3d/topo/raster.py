"""CPU 삼각형 래스터라이저 (numba). 원근 카메라, OpenCV 규약, 픽셀 중심 = (x+0.5, y+0.5)."""
import numpy as np
from numba import njit


@njit(cache=True)
def _raster(sx, sy, invz, attr, faces, H, W):
    zbuf = np.zeros((H, W), np.float32)            # 1/z, 클수록 가까움 (0 = 빈 픽셀)
    fbuf = np.full((H, W), -1, np.int64)           # 보이는 삼각형 번호
    abuf = np.zeros((H, W, attr.shape[1]), np.float32)
    for f in range(faces.shape[0]):
        a, b, c = faces[f, 0], faces[f, 1], faces[f, 2]
        if invz[a] <= 0 or invz[b] <= 0 or invz[c] <= 0:
            continue
        x0, y0, x1, y1, x2, y2 = sx[a], sy[a], sx[b], sy[b], sx[c], sy[c]
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-14:
            continue
        xmin = max(int(np.ceil(min(x0, x1, x2) - 0.5)), 0)
        xmax = min(int(np.floor(max(x0, x1, x2) - 0.5)), W - 1)
        ymin = max(int(np.ceil(min(y0, y1, y2) - 0.5)), 0)
        ymax = min(int(np.floor(max(y0, y1, y2) - 0.5)), H - 1)
        for y in range(ymin, ymax + 1):
            py = y + 0.5
            for x in range(xmin, xmax + 1):
                px = x + 0.5
                w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / den
                w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / den
                w2 = 1.0 - w0 - w1
                # 공유 모서리 위의 샘플이 빠지지 않도록 아주 작은 음수 허용
                if w0 < -1e-9 or w1 < -1e-9 or w2 < -1e-9:
                    continue
                iz = w0 * invz[a] + w1 * invz[b] + w2 * invz[c]
                if iz > zbuf[y, x]:
                    zbuf[y, x] = iz
                    fbuf[y, x] = f
                    p0 = w0 * invz[a] / iz
                    p1 = w1 * invz[b] / iz
                    p2 = 1.0 - p0 - p1
                    for k in range(attr.shape[1]):
                        abuf[y, x, k] = p0 * attr[a, k] + p1 * attr[b, k] + p2 * attr[c, k]
    return zbuf, fbuf, abuf


def render(V, tris, K, R, t, H, W, attr=None, ssaa=1):
    """V: (N,3) 월드 좌표, tris: (M,3). 반환 dict(depth, face, attr, cover) — ssaa>1이면 cover는 부분 커버리지."""
    Pc = V @ np.asarray(R).T + np.asarray(t)
    z = Pc[:, 2]
    invz = np.where(z > 1e-9, 1.0 / np.maximum(z, 1e-9), -1.0)
    Kh = np.array(K, float).copy()
    Kh[:2] *= ssaa
    sx = Kh[0, 0] * Pc[:, 0] * invz + Kh[0, 2]
    sy = Kh[1, 1] * Pc[:, 1] * invz + Kh[1, 2]
    if attr is None:
        attr = np.zeros((len(V), 1), np.float32)
    zb, fb, ab = _raster(sx, sy, invz, attr.astype(np.float32), tris.astype(np.int64), H * ssaa, W * ssaa)
    hit = zb > 0
    out = {"cover_hi": hit, "face_hi": fb, "attr_hi": ab}
    r = ssaa
    out["cover"] = hit.reshape(H, r, W, r).mean((1, 3))
    # 대표 샘플(각 블록의 가장 가까운 샘플)
    zb4 = zb.reshape(H, r, W, r).transpose(0, 2, 1, 3).reshape(H, W, r * r)
    best = zb4.argmax(-1)
    iz = np.take_along_axis(zb4, best[..., None], -1)[..., 0]
    out["depth"] = np.where(iz > 0, 1.0 / np.maximum(iz, 1e-12), 0).astype(np.float32)
    fb4 = fb.reshape(H, r, W, r).transpose(0, 2, 1, 3).reshape(H, W, r * r)
    out["face"] = np.take_along_axis(fb4, best[..., None], -1)[..., 0]
    ab4 = ab.reshape(H, r, W, r, -1).transpose(0, 2, 1, 3, 4).reshape(H, W, r * r, -1)
    out["attr"] = np.take_along_axis(ab4, best[..., None, None], 2)[:, :, 0]
    return out


def quads_to_tris(Q):
    Q = np.asarray(Q)
    return np.concatenate([Q[:, [0, 1, 2]], Q[:, [0, 2, 3]]])
