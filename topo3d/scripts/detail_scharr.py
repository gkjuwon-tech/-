"""MoGe-2 법선 + 사진 고주파(Scharr) 디테일.

n = normalize(n_base + k * (s·Gx, s·Gy, 0)),  G = Scharr(I - blur(I, σ))
파라미터(σ, k, s)는 view_000에서만 고르고 나머지 7뷰는 그대로 적용해 채점한다.
사용: python scripts/detail_scharr.py [--base work/kaggle/moge2_vitl.npz]
"""
import argparse
import itertools
import json
import os

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="work/kaggle/moge2_vitl.npz")
ap.add_argument("--out", default="results/scharr")
args = ap.parse_args()

cams = json.load(open("inputs/lucy_8v/cameras.json"))
views = cams["views"]
base = np.load(args.base)["normal"].astype(np.float32)
base /= np.linalg.norm(base, axis=-1, keepdims=True) + 1e-9

data = []
for i, v in enumerate(views):
    rgba = cv2.imread(f"inputs/lucy_8v/{v['name']}.png", cv2.IMREAD_UNCHANGED)
    I = cv2.cvtColor(rgba[..., :3], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    m = rgba[..., 3] >= 128
    ev = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    gt = np.load(f"data/lucy/turnaround_8v/{v['name']}_normal.npy") @ np.array(v["R"]).T
    data.append((I, m, ev, gt, np.nan_to_num(base[i])))


def detail(I, m, sigma):
    # 마스크 밖은 블러 값으로 채워서 실루엣 경계가 가짜 기울기를 만들지 않게
    w = cv2.GaussianBlur(m.astype(np.float32), (0, 0), sigma)
    lo = cv2.GaussianBlur(I * m, (0, 0), sigma) / np.maximum(w, 1e-3)
    hp = np.where(m, I - lo, 0).astype(np.float32)
    gx = cv2.Scharr(hp, cv2.CV_32F, 1, 0) / 32
    gy = cv2.Scharr(hp, cv2.CV_32F, 0, 1) / 32
    return gx, gy, hp


L_CAM = np.array([-0.5, -0.7, -0.5])        # 루시 렌더 키 라이트 (카메라 좌표)
L_CAM /= np.linalg.norm(L_CAM)


def apply(n, gx, gy, k, s, hp=None):
    if s == "light":                             # 밝기 고주파만큼 빛 쪽으로 기울임
        lt = L_CAM - (n @ L_CAM)[..., None] * n  # 표면 접평면에 투영한 빛 방향
        d = hp[..., None] * lt * k
    else:
        d = np.dstack([s[0] * gx, s[1] * gy, np.zeros_like(gx)]) * k
    o = n + d
    return o / (np.linalg.norm(o, axis=-1, keepdims=True) + 1e-9)


def err(n, gt, ev):
    a = np.degrees(np.arccos(np.clip((n[ev] * gt[ev]).sum(-1), -1, 1)))
    return float(np.median(a)), float((a < 11.25).mean())


# view_000에서 파라미터 탐색
I, m, ev, gt, n0 = data[0]
best = (err(n0, gt, ev)[0], None)
for sigma in (1.5, 3, 6):
    gx, gy, hp = detail(I, m, sigma)
    for s in list(itertools.product([1, -1], repeat=2)) + ["light"]:
        for k in (0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8):
            e = err(apply(n0, gx, gy, k, s, hp), gt, ev)[0]
            if e < best[0]:
                best = (e, (sigma, k, s))
print("view_000에서 고른 값 (σ, k, 부호):", best[1])

rows = []
for i, (I, m, ev, gt, n) in enumerate(data):
    e0 = err(n, gt, ev)
    if best[1]:
        sigma, k, s = best[1]
        gx, gy, hp = detail(I, m, sigma)
        e1 = err(apply(n, gx, gy, k, s, hp), gt, ev)
    else:
        e1 = e0
    rows.append((e0, e1))
    tag = "(튜닝)" if i == 0 else "(검증)"
    print(f"{views[i]['name']} {tag}  MoGe {e0[0]:6.2f}° → +Scharr {e1[0]:6.2f}°   <11.25° {e0[1]:.2f} → {e1[1]:.2f}")
print(f"검증 7뷰 평균: MoGe {np.mean([r[0][0] for r in rows[1:]]):.2f}° → +Scharr {np.mean([r[1][0] for r in rows[1:]]):.2f}°")

# 얼굴 비교 (옆 조명)
os.makedirs(args.out, exist_ok=True)
RL = np.array([0.8, -0.1, -0.6])
RL /= np.linalg.norm(RL)
I, m, ev, gt, n = data[0]
ys, xs = np.nonzero(m)
y0, cx = ys.min(), (xs.min() + xs.max()) // 2
tiles = []
cands = [(gt, "GT"), (n, "MoGe-2")]
if best[1]:
    sigma, k, s = best[1]
    gx, gy, hp = detail(I, m, sigma)
    cands.append((apply(n, gx, gy, k, s, hp), f"MoGe-2 + detail ({s})"))
for nn, t in cands:
    sh = 0.15 + 0.85 * np.clip(nn @ RL, 0, 1)
    sh[~m] = 0.4
    c = (sh * 255).astype(np.uint8)[y0 + 58:y0 + 278, cx - 150:cx + 150].copy()
    cv2.putText(c, t, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
    tiles.append(c)
cv2.imwrite(os.path.join(args.out, "face.png"), cv2.resize(np.hstack(tiles), None, fx=2, fy=2,
                                                            interpolation=cv2.INTER_NEAREST))
