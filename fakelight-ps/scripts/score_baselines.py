"""노멀 직접 예측 모델(Marigold, StableNormal, DSINE) 결과를 채점한다.

모델마다 좌표계 부호(x/y/z 방향)가 다르다. 정답을 보고 세트마다 부호를 고르면 반칙이므로,
대조군(render)에서 8가지 부호 조합 중 가장 잘 맞는 것을 한 번 정하고, 그 규약을 jugg 세트에 그대로 쓴다.

사용법:
    python scripts/score_baselines.py results/normal_baselines --out results/normal_baselines/LEADERBOARD.md
"""
import argparse
import glob
import itertools
import os

import numpy as np
from PIL import Image

from ps_eval import load_mask, mean_angular_error

GT = {
    "jugg": ("renders/bunny_front_aligned_768/normal_gt.npy", "renders/bunny_front_aligned_768/mask.png"),
    "render": ("renders/bunny_front_normal_gt.npy", "renders/bunny_front_mask.png"),
}
FLIPS = [np.array(f) for f in itertools.product([1, -1], repeat=3)]


def load_pred(path, shape):
    n = np.load(path).astype(np.float32)
    if n.shape[:2] != shape:
        n = np.stack([np.asarray(Image.fromarray(n[..., c], "F").resize((shape[1], shape[0]), Image.BILINEAR))
                      for c in range(3)], 2)
    return n / (np.linalg.norm(n, axis=2, keepdims=True) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gts = {k: (np.load(g), None) for k, (g, _) in GT.items()}
    gts = {k: (g, load_mask(GT[k][1], g.shape[:2])) for k, (g, _) in gts.items()}
    models = sorted({os.path.basename(p).split("__")[0] for p in glob.glob(os.path.join(args.results, "*__*.npy"))})

    rows = []
    for model in models:
        g, m = gts["render"]
        pr = load_pred(os.path.join(args.results, f"{model}__render.npy"), g.shape[:2])
        errs = [mean_angular_error(pr * f, g, m)[0] for f in FLIPS]
        flip = FLIPS[int(np.argmin(errs))]
        for subject in ("render", "jugg"):
            g, m = gts[subject]
            p = load_pred(os.path.join(args.results, f"{model}__{subject}.npy"), g.shape[:2]) * flip
            rows.append((subject, model, mean_angular_error(p, g, m)[0], tuple(int(x) for x in flip)))

    table = "\n".join(["| 입력 | 모델 | 평균 각도 오차 | 부호 규약 (render로 결정) |", "|---|---|---|---|"]
                      + [f"| {s} | {mo} | **{e:.2f}°** | {f} |" for s, mo, e, f in rows])
    print(table)
    if args.out:
        open(args.out, "w").write("# 노멀 직접 예측 기준선 (버니 정면)\n\n" + table + "\n")


if __name__ == "__main__":
    main()
