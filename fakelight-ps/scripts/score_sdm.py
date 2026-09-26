"""SDM-UniPS 결과 폴더(sdm_results/*)를 전부 채점하고, 시드별 결과의 앙상블도 채점한다.

세트 이름의 앞부분으로 정답을 고른다:
  jugg_*   → MV-Adapter 좌표계로 정렬한 정답 (renders/bunny_front_aligned_768)
  render_* → 원본 렌더 좌표계 정답
  real_*   → 진짜 조명 렌더 정답

앙상블: {subject}_s{seed} 세트들의 노멀을 픽셀별로 평균 후 재정규화.

사용법:
    python scripts/score_sdm.py results/relight_ps_v2 --out results/relight_ps_v2/LEADERBOARD.md
"""
import argparse
import glob
import os
import re

import numpy as np
from PIL import Image

from ps_eval import load_mask, load_normal_png, mean_angular_error

GT = {
    "jugg": ("renders/bunny_front_aligned_768/normal_gt.npy", "renders/bunny_front_aligned_768/mask.png"),
    "render": ("renders/bunny_front_normal_gt.npy", "renders/bunny_front_mask.png"),
    "real": ("renders/bunny_real_lights.data/normal_gt.npy", "renders/bunny_real_lights.data/mask.png"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows, per_seed = [], {}
    for path in sorted(glob.glob(os.path.join(args.results, "sdm_results", "*", "normal.png"))):
        name = os.path.basename(os.path.dirname(path)).removesuffix(".data")
        subject = name.split("_")[0]
        gt = np.load(GT[subject][0])
        mask = load_mask(GT[subject][1], gt.shape[:2])
        pred = load_normal_png(path, gt.shape[:2])
        rows.append((name, mean_angular_error(pred, gt, mask)[0]))
        if re.fullmatch(rf"{subject}_s\d+", name):
            per_seed.setdefault(subject, []).append(pred)

    for subject, preds in per_seed.items():
        gt = np.load(GT[subject][0])
        mask = load_mask(GT[subject][1], gt.shape[:2])
        n = np.mean(preds, 0)
        n /= np.linalg.norm(n, axis=2, keepdims=True) + 1e-12
        rows.append((f"{subject}_seed_ensemble({len(preds)})", mean_angular_error(n, gt, mask)[0]))

    rows.sort(key=lambda r: (r[0].split("_")[0], r[1]))
    table = "\n".join(["| 세트 | SDM-UniPS 평균 각도 오차 |", "|---|---|"]
                      + [f"| {n} | **{e:.2f}°** |" for n, e in rows])
    print(table)
    if args.out:
        open(args.out, "w").write("# 버니 정면 노멀 기록판 (SDM-UniPS)\n\n" + table + "\n")


if __name__ == "__main__":
    main()
