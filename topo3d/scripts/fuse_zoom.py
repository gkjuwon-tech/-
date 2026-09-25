"""확대 크롭 MoGe-2 타일을 합쳐 뷰별 법선을 만들고 채점한다.

1) 타일 법선을 전체 이미지 MoGe 법선에 회전 정렬 (Kabsch, 정답 미사용)
2) Hann 창으로 겹침 영역을 섞음 → n_zoom
3) 큰 형태는 전체 이미지(n_full의 저주파), 디테일은 n_zoom의 고주파
사용: python scripts/fuse_zoom.py [--zoom work/kaggle_zoom/moge2_zoom.npz]
"""
import argparse
import json
import os

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--zoom", default="work/kaggle_zoom/moge2_zoom.npz")
ap.add_argument("--full", default="work/kaggle/moge2_vitl.npz")
ap.add_argument("--sigma", type=float, default=8.0, help="저주파/고주파 경계 (px)")
ap.add_argument("--out", default="work/moge_zoom_fused.npz")
args = ap.parse_args()

cams = json.load(open("inputs/lucy_8v/cameras.json"))
views = cams["views"]
H, W = cams["height"], cams["width"]
Z = np.load(args.zoom)
full = np.nan_to_num(np.load(args.full)["normal"].astype(np.float32))
full /= np.linalg.norm(full, axis=-1, keepdims=True) + 1e-9


def unit(n):
    return n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9)


def kabsch(src, dst):
    U, _, Vt = np.linalg.svd(src.T @ dst)
    d = np.sign(np.linalg.det(U @ Vt))
    return U @ np.diag([1, 1, d]) @ Vt            # src @ R ≈ dst


def lowpass(n, m, s):
    w = cv2.GaussianBlur(m.astype(np.float32), (0, 0), s)
    return np.dstack([cv2.GaussianBlur(n[..., k] * m, (0, 0), s) for k in range(3)]) / np.maximum(w, 1e-4)[..., None]


def err(n, gt, ev):
    a = np.degrees(np.arccos(np.clip((n[ev] * gt[ev]).sum(-1), -1, 1)))
    return float(np.median(a))


RLs = [np.array(v) / np.linalg.norm(v) for v in ([0.8, -0.1, -0.6], [-0.7, 0.3, -0.6], [0.1, 0.8, -0.6])]


def detail_corr(n, gt, ev):
    cs = []
    for RL in RLs:
        a = np.clip(n @ RL, 0, 1).astype(np.float32)
        b = np.clip(gt @ RL, 0, 1).astype(np.float32)
        ha, hb = a - cv2.GaussianBlur(a, (0, 0), 3), b - cv2.GaussianBlur(b, (0, 0), 3)
        cs.append(np.corrcoef(ha[ev], hb[ev])[0, 1])
    return float(np.mean(cs))


T = int(Z["boxes"][0][2])
hann = np.outer(np.hanning(T), np.hanning(T)).astype(np.float32) + 1e-3
fused_all, zoom_all = [], []
print("view      full   zoom   fused    디테일상관 full→zoom→fused")
for vi, v in enumerate(views):
    a = cv2.imread(f"inputs/lucy_8v/{v['name']}.png", cv2.IMREAD_UNCHANGED)[..., 3] >= 128
    ev = cv2.erode(a.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    gt = np.load(f"data/lucy/turnaround_8v/{v['name']}_normal.npy") @ np.array(v["R"]).T
    acc = np.zeros((H, W, 3), np.float32)
    wacc = np.zeros((H, W), np.float32)
    for (y0, x0, t), nt in zip(Z["boxes"][Z["views"] == vi], Z["normals"][Z["views"] == vi]):
        nt = unit(np.nan_to_num(nt.astype(np.float32)))
        sl = (slice(y0, y0 + t), slice(x0, x0 + t))
        mm = a[sl] & np.isfinite(nt).all(-1)
        if mm.sum() < 50:
            continue
        R = kabsch(nt[mm], full[vi][sl][mm])
        acc[sl] += (nt @ R) * (hann * mm)[..., None]
        wacc[sl] += hann * mm
    nz = unit(np.where(wacc[..., None] > 1e-6, acc / np.maximum(wacc, 1e-6)[..., None], full[vi]))
    nf = unit(lowpass(full[vi], a, args.sigma) + nz - lowpass(nz, a, args.sigma))
    zoom_all.append(nz)
    fused_all.append(nf)
    print(f"{v['name']}  {err(full[vi], gt, ev):5.2f}  {err(nz, gt, ev):5.2f}  {err(nf, gt, ev):5.2f}     "
          f"{detail_corr(full[vi], gt, ev):.3f} → {detail_corr(nz, gt, ev):.3f} → {detail_corr(nf, gt, ev):.3f}")
np.savez_compressed(args.out, normal=np.array(fused_all, np.float16), zoom=np.array(zoom_all, np.float16))

# 얼굴 비교
os.makedirs("results/zoom", exist_ok=True)
a = cv2.imread("inputs/lucy_8v/view_000.png", cv2.IMREAD_UNCHANGED)[..., 3] >= 128
gt = np.load("data/lucy/turnaround_8v/view_000_normal.npy") @ np.array(views[0]["R"]).T
ys, xs = np.nonzero(a)
y0, cx = ys.min(), (xs.min() + xs.max()) // 2
tiles = []
for n, t in [(gt, "GT"), (full[0], "MoGe-2 full"), (zoom_all[0], "MoGe-2 zoom x3"), (fused_all[0], "full low + zoom high")]:
    s = 0.15 + 0.85 * np.clip(n @ RLs[0], 0, 1)
    s[~a] = 0.4
    c = (s * 255).astype(np.uint8)[y0 + 58:y0 + 278, cx - 150:cx + 150].copy()
    cv2.putText(c, t, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
    tiles.append(c)
cv2.imwrite("results/zoom/face.png", cv2.resize(np.hstack(tiles), None, fx=1.6, fy=1.6, interpolation=cv2.INTER_AREA))
