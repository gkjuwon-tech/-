"""DiLiGenT 원정 채점: 커널 출력(out/{OBJ}_{IMG})에서 버니 레시피를 그대로 마무리하고 정답과 비교한다.

  1) SDM-UniPS 노멀 3개(ic_rot, ic_plain, nlr_all)를 합체:
       ic = avg(ic_rot, ic_plain), 저주파 = blur(avg(ic, nlr), σ=4), 고주파 = ic - blur(ic, σ=4)
  2) 무늬 인식 디테일 보정: IC-Light 32장으로 알베도 추정 → 입력 크롭을 알베도로 나눔 → 음영 기반 보정 (λ=0.1, μ=0.02)
  3) 크롭 좌표 → 원본 612×512로 되돌려서 Normal_gt.mat과 마스크 안에서 평균 각도 오차
단계별 점수와 같은 사진으로 돌린 RoSE 점수를 함께 기록한다.

사용법:
    python scripts/diligent_eval.py results/diligent/ballPNG_025 --data data/DiLiGenT/pmsData
"""
import argparse
import glob
import json
import os

import numpy as np
import scipy.io as sio
from PIL import Image

from detail_metrics import detail_scores, masked_blur
from ps_eval import load_normal_png, mean_angular_error
from shading_refine import estimate_albedo, refine

SIGMA, LAM, MU = 4, 0.1, 0.02
norm = lambda n: n / (np.linalg.norm(n, axis=2, keepdims=True) + 1e-12)


def resize_normal(n, size):
    out = np.stack([np.asarray(Image.fromarray(n[..., c].astype(np.float32), "F").resize((size, size), Image.BILINEAR))
                    for c in range(3)], 2)
    return norm(out)


def uncrop(n_crop, crop, shape):
    """크롭(frame×frame) 노멀을 원본 좌표로. 크롭 밖은 0."""
    n = resize_normal(n_crop, crop["size"])
    out = np.zeros(tuple(shape) + (3,))
    y0, x0, s = crop["y0"], crop["x0"], crop["size"]
    ys, xs = max(0, y0), max(0, x0)
    ye, xe = min(shape[0], y0 + s), min(shape[1], x0 + s)
    out[ys:ye, xs:xe] = n[ys - y0:ye - y0, xs - x0:xe - x0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--data", default="data/DiLiGenT/pmsData")
    args = ap.parse_args()

    crop = json.load(open(os.path.join(args.folder, "crop.json")))
    F = crop["frame"]
    obj_dir = os.path.join(args.data, crop["obj"])
    gt = sio.loadmat(os.path.join(obj_dir, "Normal_gt.mat"))["Normal_gt"].astype(np.float64)
    gmask = np.asarray(Image.open(os.path.join(obj_dir, "mask.png")).convert("L")) > 127
    cm = np.asarray(Image.open(os.path.join(args.folder, "mask_crop.png")).convert("L")) > 127

    sdm = lambda s: load_normal_png(os.path.join(args.folder, "sdm_results", f"{s}.data", "normal.png"), (F, F))
    n_rot, n_plain, n_nlr = sdm("ic_rot"), sdm("ic_plain"), sdm("nlr_all")
    ic = norm(n_rot + n_plain)
    fused = norm(masked_blur(norm(ic + n_nlr), cm, SIGMA) + (ic - masked_blur(ic, cm, SIGMA)))

    inp = np.asarray(Image.open(os.path.join(args.folder, "input_crop.png")).convert("L"), dtype=np.float64) / 255
    rl = np.stack([np.asarray(Image.open(p).convert("L"), dtype=np.float64) / 255
                   for p in sorted(glob.glob(os.path.join(args.folder, "sets", "ic_*__L_*.png")))])
    rho = estimate_albedo(rl, fused, cm)
    refined, _ = refine(np.where(cm, inp / np.clip(rho, 0.05, None), inp), fused, cm, LAM, MU)

    rows = {}
    stages = {"IC-Light 회전 트릭": n_rot, "IC-Light 회전 없음": n_plain, "Neural LightRig": n_nlr,
              "합체": fused, "합체 + 디테일 보정 (최종)": refined}
    for name, n in stages.items():
        full = uncrop(n * cm[..., None], crop, gmask.shape)
        valid = gmask & (np.linalg.norm(full, axis=2) > 0.5)
        rows[name] = (mean_angular_error(full, gt, gmask)[0], detail_scores(full, gt, gmask)[0], valid.sum() / gmask.sum())
    rose_path = os.path.join(args.folder, "rose_normal.npy")
    if os.path.exists(rose_path):
        full = uncrop(resize_normal(np.load(rose_path), F) * cm[..., None], crop, gmask.shape)
        rows["RoSE (같은 사진)"] = (mean_angular_error(full, gt, gmask)[0], detail_scores(full, gt, gmask)[0], 1.0)

    final = uncrop(refined * cm[..., None], crop, gmask.shape)
    Image.fromarray(((final * 0.5 + 0.5) * 255 * gmask[..., None]).astype(np.uint8)).save(
        os.path.join(args.folder, "final_normal.png"))
    lines = [f"# {crop['obj']} (사진 {crop['img']}.png)", "", "| 단계 | 평균 각도 오차 | 디테일 상관 | 커버리지 |", "|---|---|---|---|"]
    lines += [f"| {k} | **{v[0]:.2f}°** | {v[1]:.3f} | {v[2]:.0%} |" for k, v in rows.items()]
    text = "\n".join(lines)
    print(text)
    open(os.path.join(args.folder, "RESULT.md"), "w").write(text + "\n")
    json.dump({k: v[0] for k, v in rows.items()}, open(os.path.join(args.folder, "scores.json"), "w"), indent=2,
              ensure_ascii=False)


if __name__ == "__main__":
    main()
