"""모델 법선/깊이를 루시 정답과 비교 (8뷰).

사용: python scripts/audit_normals.py work/kaggle/moge2_vitl.npz [...]
- normal: (N,H,W,3). 축 부호 규약은 view_000에서 8가지 조합 중 최적을 골라 전 뷰에 적용
- depth(+crop): 카메라 z 깊이. 정답과 스케일·시프트 맞춘 뒤 상관·오차, 그리고 깊이에서 계산한 법선 오차
"""
import itertools
import json
import sys

import cv2
import numpy as np

cams = json.load(open("inputs/lucy_8v/cameras.json"))
views = cams["views"]
H, W = cams["height"], cams["width"]


def gt(i):
    v = views[i]
    R = np.array(v["R"])
    n = np.load(f"data/lucy/turnaround_8v/{v['name']}_normal.npy") @ R.T
    d = np.load(f"data/lucy/turnaround_8v/{v['name']}_depth.npy")
    a = cv2.imread(f"inputs/lucy_8v/{v['name']}.png", cv2.IMREAD_UNCHANGED)[..., 3] >= 128
    ev = cv2.erode(a.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    return n, d, ev


def ang(n, g, e):
    n = n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9)
    ok = e & np.isfinite(n).all(-1)
    a = np.degrees(np.arccos(np.clip((n[ok] * g[ok]).sum(-1), -1, 1)))
    return np.median(a), (a < 11.25).mean()


def depth_normals(d, K):
    yy, xx = np.mgrid[0:d.shape[0], 0:d.shape[1]] + 0.5
    P = np.dstack([(xx - K[0, 2]) / K[0, 0] * d, (yy - K[1, 2]) / K[1, 1] * d, d])
    dx = np.zeros_like(P)
    dy = np.zeros_like(P)
    dx[:, 1:-1] = (P[:, 2:] - P[:, :-2]) / 2
    dy[1:-1] = (P[2:] - P[:-2]) / 2
    return np.cross(dy, dx)


for path in sys.argv[1:]:
    r = dict(np.load(path))
    print(f"== {path}")
    if "normal" in r:
        Nn = r["normal"].astype(np.float32)
        g0, _, e0 = gt(0)
        n0 = cv2.resize(Nn[0], (W, H)) if Nn[0].shape[:2] != (H, W) else Nn[0]
        best = min(itertools.product([1, -1], repeat=3), key=lambda s: ang(n0 * np.array(s), g0, e0)[0])
        print(f"  법선 축 부호 {best}")
        rows = []
        for i in range(len(views)):
            g, _, e = gt(i)
            n = cv2.resize(Nn[i], (W, H)) if Nn[i].shape[:2] != (H, W) else Nn[i]
            rows.append(ang(n * np.array(best), g, e))
        for v, (md, fr) in zip(views, rows):
            print(f"  {v['name']} 법선 {md:6.2f}°  <11.25° {fr:.2f}")
        print(f"  평균 중앙값 {np.mean([x[0] for x in rows]):.2f}°")
    if "depth" in r:
        crop = r.get("crop")
        rows = []
        for i in range(len(views)):
            g, gd, e = gt(i)
            K = np.array(views[i]["K"])
            d = r["depth"][i].astype(np.float64)
            if crop is not None:
                y0, y1, x0, x1 = [int(c) for c in crop]
                g, gd, e = g[y0:y1, x0:x1], gd[y0:y1, x0:x1], e[y0:y1, x0:x1]
                K = K.copy()
                K[0, 2] -= x0
                K[1, 2] -= y0
            if d.shape != gd.shape:
                d = cv2.resize(d, gd.shape[::-1], interpolation=cv2.INTER_LINEAR)
            ok = e & np.isfinite(d) & (d > 0)
            A = np.stack([d[ok], np.ones(ok.sum())], 1)
            s, b = np.linalg.lstsq(A, gd[ok], rcond=None)[0]
            err = np.median(np.abs(s * d[ok] + b - gd[ok])) / (gd[ok].max() - gd[ok].min())
            md, _ = ang(depth_normals(s * d + b, K), g, e)
            rows.append((np.corrcoef(d[ok], gd[ok])[0, 1], err, md))
        for v, (c, er, md) in zip(views, rows):
            print(f"  {v['name']} 깊이 상관 {c:.3f}  오차/폭 {er:.3f}  깊이→법선 {md:6.2f}°")
        print(f"  평균 깊이→법선 {np.mean([x[2] for x in rows]):.2f}°")
