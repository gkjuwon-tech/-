"""빛 균일화 보정: IC-Light 이미지에서 '위치에 따라 변하는 빛 세기와 환경광'을 걷어내 먼 곳의 균일한 빛처럼 만든다.

진단 결과 IC-Light는 배경 그라데이션 쪽이 더 밝게 비춰지는 '가까운 조명'처럼 행동한다
(빛 세기가 위치에 따라 변한다고 보면 R² 0.747 → 0.851). 광도 스테레오는 먼 곳의 균일한 빛을 가정하므로,
초기 노멀(SDM-UniPS 결과, 정답 아님)로 이미지마다
    I_lin(x,y) ≈ s_k(x,y)·max(0, n·g_k) + a_k(x,y),   s_k = 1 + αX + βY,  a_k = a0 + a1X + a2Y
를 맞추고,  I_corr = (I_lin - a_k(x,y)) / s_k(x,y) + a0  로 보정해서 16비트 선형 PNG로 저장한다.

사용법:
    python scripts/correct_lighting.py --images "results/relight_ps_v2/jugg_s1.data/L_*.png" \
        --init results/relight_ps_v2/sdm_results/jugg_s1_x2.data/normal.png \
        --mask results/relight_ps_v2/jugg_s1.data/mask.png --out build/corr/jugg_s1_ramp --prefix jugg_s1_ramp
"""
import argparse
import glob
import os
import shutil

import cv2
import numpy as np
from PIL import Image

from physics_refine import fit_lights, srgb2lin
from ps_eval import load_mask, load_normal_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--out", required=True, help="출력 폴더 (평평하게 {prefix}__L_XX.png)")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--no-ramp", action="store_true", help="선형화만 (비교용)")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.images))
    first = Image.open(paths[0])
    shape = (first.height, first.width)
    m = load_mask(args.mask, shape)
    n0 = load_normal_png(args.init, shape)
    lin = np.stack([srgb2lin(np.asarray(Image.open(p).convert("RGB"), dtype=np.float64) / 255.0) for p in paths])
    gray = lin.mean(3)

    os.makedirs(args.out, exist_ok=True)
    if args.no_ramp:
        corr = lin
    else:
        yy, xx = np.nonzero(m)
        XY = ((xx / shape[1] - 0.5) * 2, -(yy / shape[0] - 0.5) * 2)
        G, a_m, s_m = fit_lights(gray[:, m], n0[m], XY)
        # 마스크 안에서 맞춘 선형 계수를 화면 전체로 확장
        H, W = shape
        Yf, Xf = np.mgrid[0:H, 0:W]
        Xf, Yf = (Xf / W - 0.5) * 2, -(Yf / H - 0.5) * 2
        corr = np.empty_like(lin)
        for k in range(len(paths)):
            A = np.c_[np.ones(m.sum()), XY[0], XY[1]]
            ca, *_ = np.linalg.lstsq(A, a_m[k], rcond=None)
            cs, *_ = np.linalg.lstsq(A, s_m[k], rcond=None)
            a_f = ca[0] + ca[1] * Xf + ca[2] * Yf
            s_f = np.clip(cs[0] + cs[1] * Xf + cs[2] * Yf, 0.2, 5)
            corr[k] = np.clip((lin[k] - a_f[..., None]) / s_f[..., None] + ca[0], 0, None)
            print(f"{os.path.basename(paths[k])}: 세기 기울기 α={cs[1]:+.2f} β={cs[2]:+.2f}  "
                  f"환경광 {ca[0]:.3f} ({ca[1]:+.3f}, {ca[2]:+.3f})")
    corr /= corr.max() + 1e-8
    for k, im in enumerate(corr):
        cv2.imwrite(os.path.join(args.out, f"{args.prefix}__L_{k:02d}.png"),
                    (im * 65535).astype(np.uint16)[..., ::-1])
    shutil.copy(args.mask, os.path.join(args.out, f"{args.prefix}__mask.png"))
    print(f"saved {len(paths)} images to {args.out}")


if __name__ == "__main__":
    main()
