"""광도 스테레오 풀이기 + 노멀 채점기.

- solve_ols: 조명 방향을 아는 경우의 고전 최소제곱 (Woodham 1980). 그림자 진 관측은 가중치 0.
- mean_angular_error: 마스크 안에서 정답 노멀과의 평균 각도 오차(°)
- load_normal_png: SDM-UniPS/DiLiGenT 형식 노멀 PNG ((n+1)/2 * 255, RGB) 읽기

사용법:
    # 진짜 조명 폴더를 최소제곱으로 풀고 채점
    python scripts/ps_eval.py ols renders/bunny_real_lights.data
    # 노멀 PNG 채점
    python scripts/ps_eval.py score pred_normal.png --gt gt.npy --mask mask.png
"""
import argparse
import glob
import json
import os

import numpy as np
from PIL import Image


def load_gray(path):
    im = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    return im.mean(2)


def load_mask(path, shape=None):
    m = Image.open(path).convert("L")
    if shape is not None and m.size != (shape[1], shape[0]):
        m = m.resize((shape[1], shape[0]), Image.NEAREST)
    return np.asarray(m) > 127


def load_normal_png(path, shape=None):
    im = Image.open(path).convert("RGB")
    if shape is not None and im.size != (shape[1], shape[0]):
        im = im.resize((shape[1], shape[0]), Image.BILINEAR)
    n = np.asarray(im, dtype=np.float64) / 255.0 * 2 - 1
    return n / (np.linalg.norm(n, axis=2, keepdims=True) + 1e-12)


def solve_ols(images, lights, shadow_thresh=0.02):
    """images: (K, H, W) 밝기, lights: (K, 3). 반환: 노멀 (H, W, 3), 알베도 (H, W).

    픽셀마다 그림자가 아닌 관측(>thresh)만 써서 가중 최소제곱으로 ρn을 푼다.
    """
    K, H, W = images.shape
    I = images.reshape(K, -1)
    w = (I > shadow_thresh).astype(np.float64)
    # 픽셀별 3x3 정규방정식: (Lᵀ W L) g = Lᵀ W I
    A = np.einsum("kp,ki,kj->pij", w, lights, lights)
    b = np.einsum("kp,ki,kp->pi", w, lights, I)
    ok = w.sum(0) >= 3
    g = np.zeros((H * W, 3))
    g[ok] = np.linalg.solve(A[ok] + 1e-9 * np.eye(3), b[ok][..., None])[..., 0]
    albedo = np.linalg.norm(g, axis=1)
    n = g / (albedo[:, None] + 1e-12)
    return n.reshape(H, W, 3), albedo.reshape(H, W)


def mean_angular_error(pred, gt, mask):
    cos = np.clip((pred * gt).sum(2), -1, 1)
    err = np.degrees(np.arccos(cos))
    return float(err[mask].mean()), err


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("ols")
    o.add_argument("folder")
    s = sub.add_parser("score")
    s.add_argument("pred")
    s.add_argument("--gt", required=True, help=".npy (H,W,3)")
    s.add_argument("--mask", required=True)
    args = ap.parse_args()

    if args.cmd == "ols":
        paths = sorted(glob.glob(os.path.join(args.folder, "L_*.png")))
        images = np.stack([load_gray(p) for p in paths])
        lights = np.array(json.load(open(os.path.join(args.folder, "lights.json")))["lights"])
        gt = np.load(os.path.join(args.folder, "normal_gt.npy"))
        mask = load_mask(os.path.join(args.folder, "mask.png"))
        n, _ = solve_ols(images, lights)
        mae, _ = mean_angular_error(n, gt, mask)
        Image.fromarray(((n * 0.5 + 0.5) * 255 * mask[..., None]).astype(np.uint8)).save(
            os.path.join(args.folder, "normal_ols.png"))
        print(f"OLS ({len(paths)} lights) mean angular error: {mae:.2f} deg")
    else:
        gt = np.load(args.gt)
        mask = load_mask(args.mask, gt.shape[:2])
        pred = load_normal_png(args.pred, gt.shape[:2])
        mae, _ = mean_angular_error(pred, gt, mask)
        print(f"{args.pred}: {mae:.2f} deg")


if __name__ == "__main__":
    main()
