"""relight_ps 커널 결과를 정답 노멀로 채점해서 기록판(markdown)을 만든다.

세트:
  real_lights    : 진짜 조명 렌더 8장 (대조군, 원본 렌더 좌표)
  render_iclight : 원본 렌더를 IC-Light로 가짜 조명 8장 (원본 렌더 좌표)
  jugg_iclight   : MV-Adapter+Juggernaut 정면 뷰를 IC-Light로 가짜 조명 8장 (MV-Adapter 좌표)

풀이기:
  SDM-UniPS           : 조명 방향 필요 없음 (커널에서 실행)
  OLS(추정 방향)       : IC-Light 배경 그라데이션 방향을 극각 --polar로 가정한 최소제곱
  IC-Light 합치기 공식 : IC-Light 데모의 좌/우/위/아래 나눗셈 휴리스틱
  평평한 노멀          : 모든 픽셀 (0,0,1). 이것보다 나쁘면 의미 없는 결과

사용법:
    python scripts/score_relight.py results/relight_ps --out results/relight_ps/LEADERBOARD.md
"""
import argparse
import glob
import json
import os

import numpy as np
from PIL import Image

from ps_eval import load_gray, load_mask, load_normal_png, mean_angular_error, solve_ols

GT = {
    "real_lights": ("renders/bunny_real_lights.data/normal_gt.npy", "renders/bunny_real_lights.data/mask.png"),
    "render_iclight": ("renders/bunny_front_normal_gt.npy", "renders/bunny_front_mask.png"),
    "jugg_iclight": ("renders/bunny_front_aligned_768/normal_gt.npy", "renders/bunny_front_aligned_768/mask.png"),
}


def iclight_merge(imgs_by_angle):
    """IC-Light gradio_demo_bg.process_normal과 같은 공식 (0=오른쪽, 90=위, 180=왼쪽, 270=아래)."""
    right, top, left, bottom = (imgs_by_angle[a] for a in (0, 90, 180, 270))
    ambient = (left + right + bottom + top) / 4.0
    div = lambda a: (a + 1e-5) / (ambient + 1e-5) - 1.0
    u = (div(right) - div(left)) * 0.5
    v = (div(top) - div(bottom)) * 0.5
    h = np.clip(1.0 - u ** 2 - v ** 2, 0, 1e5) ** (0.5 * 10.0)
    n = np.stack([u, v, h], 2)
    return n / (np.linalg.norm(n, axis=2, keepdims=True) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--polar", type=float, default=45.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    for name, (gt_path, mask_path) in GT.items():
        gt = np.load(gt_path)
        mask = load_mask(mask_path, gt.shape[:2])
        H, W = gt.shape[:2]

        flat = np.zeros_like(gt); flat[..., 2] = 1
        rows.append((name, "평평한 노멀 (기준선)", mean_angular_error(flat, gt, mask)[0], 1.0))

        sdm = os.path.join(args.results, "sdm_results", f"{name}.data", "normal.png")
        if os.path.exists(sdm):
            pred = load_normal_png(sdm, (H, W))
            valid = mask & (np.asarray(Image.open(sdm).convert("L").resize((W, H))) > 0)
            rows.append((name, "SDM-UniPS", mean_angular_error(pred, gt, valid)[0], valid.sum() / mask.sum()))

        folder = os.path.join(args.results, f"{name}.data")
        meta = os.path.join(folder, "relight.json")
        if os.path.exists(meta):
            angles = json.load(open(meta))["angles"]
            imgs = np.stack([load_gray(p) for p in sorted(glob.glob(os.path.join(folder, "L_*.png")))])
            if imgs.shape[1:] != (H, W):
                imgs = np.stack([np.asarray(Image.fromarray(i.astype(np.float32), "F").resize((W, H))) for i in imgs])
            a, phi = np.deg2rad(args.polar), np.deg2rad(angles)
            L = np.stack([np.sin(a) * np.cos(phi), np.sin(a) * np.sin(phi), np.full(len(phi), np.cos(a))], 1)
            n_ols, _ = solve_ols(imgs, L)
            rows.append((name, f"OLS (방향 추정, 극각 {args.polar:.0f}°)", mean_angular_error(n_ols, gt, mask)[0], 1.0))
            by_angle = {round(t) % 360: im for t, im in zip(angles, imgs)}
            if all(k in by_angle for k in (0, 90, 180, 270)):
                n_ic = iclight_merge(by_angle)
                rows.append((name, "IC-Light 합치기 공식", mean_angular_error(n_ic, gt, mask)[0], 1.0))

    lines = ["| 세트 | 풀이기 | 평균 각도 오차 | 커버리지 |", "|---|---|---|---|"]
    lines += [f"| {s} | {m} | **{e:.2f}°** | {c:.0%} |" for s, m, e, c in rows]
    table = "\n".join(lines)
    print(table)
    if args.out:
        open(args.out, "w").write("# 버니 정면 노멀 기록판\n\n" + table + "\n")


if __name__ == "__main__":
    main()
