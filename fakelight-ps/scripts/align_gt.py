"""정답 노멀/마스크를 MV-Adapter 출력 좌표계에 맞춘다.

MV-Adapter I2MV는 입력 RGBA를 알파 bbox로 자르고, 긴 변을 출력 크기의 0.9로 줄여 가운데 둔다
(scripts/inference_i2mv_sdxl.py의 preprocess_image). 생성된 정면(0°) 뷰는 이 틀을 따르므로
정답에도 같은 변환을 적용해야 픽셀 단위로 채점할 수 있다.

생성 뷰의 실루엣(배경색과 다른 픽셀)과 정렬된 정답 마스크의 IoU도 출력해서 정렬이 맞는지 확인한다.

사용법:
    python scripts/align_gt.py --view results/audition/Juggernaut-XL-v9/view_az000.png \
        --out renders/bunny_front_aligned_768
"""
import argparse
import os

import numpy as np
from PIL import Image


def mvadapter_frame(alpha, size):
    """preprocess_image와 같은 bbox/크기/패딩 계산. 반환: (y0, y1, x0, x1, H, W, top, left)"""
    Hs, Ws = alpha.shape
    y, x = np.where(alpha)
    y0, y1 = max(y.min() - 1, 0), min(y.max() + 1, Hs)
    x0, x1 = max(x.min() - 1, 0), min(x.max() + 1, Ws)
    H, W = y1 - y0, x1 - x0
    if H > W:
        W = int(W * (size * 0.9) / H)
        H = int(size * 0.9)
    else:
        H = int(H * (size * 0.9) / W)
        W = int(size * 0.9)
    return y0, y1, x0, x1, H, W, (size - H) // 2, (size - W) // 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgba", default="renders/bunny_front_rgba.png")
    ap.add_argument("--normal", default="renders/bunny_front_normal_gt.npy")
    ap.add_argument("--view", default=None, help="정렬 확인용 생성 정면 뷰")
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--out", default="renders/bunny_front_aligned_768")
    args = ap.parse_args()

    rgba = np.asarray(Image.open(args.rgba))
    alpha = rgba[..., 3] > 0
    normal = np.load(args.normal)
    y0, y1, x0, x1, H, W, top, left = mvadapter_frame(alpha, args.size)

    crop_n = normal[y0:y1, x0:x1]
    crop_m = alpha[y0:y1, x0:x1]
    # 노멀은 채널별로 bilinear 리사이즈 후 재정규화, 마스크는 nearest
    ch = [np.asarray(Image.fromarray(crop_n[..., c].astype(np.float32), "F").resize((W, H), Image.BILINEAR))
          for c in range(3)]
    n_rs = np.stack(ch, 2)
    m_rs = np.asarray(Image.fromarray(crop_m.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 127
    n_rs /= np.linalg.norm(n_rs, axis=2, keepdims=True) + 1e-12

    n_out = np.zeros((args.size, args.size, 3), np.float32)
    m_out = np.zeros((args.size, args.size), bool)
    n_out[top:top + H, left:left + W] = n_rs
    m_out[top:top + H, left:left + W] = m_rs
    n_out[~m_out] = 0

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "normal_gt.npy"), n_out)
    Image.fromarray((m_out * 255).astype(np.uint8)).save(os.path.join(args.out, "mask.png"))
    Image.fromarray(((n_out * 0.5 + 0.5) * 255 * m_out[..., None]).astype(np.uint8)).save(
        os.path.join(args.out, "normal_gt.png"))
    print(f"saved aligned GT to {args.out}")

    if args.view:
        v = np.asarray(Image.open(args.view).convert("RGB"), dtype=np.float64)
        bg = np.median(np.concatenate([v[:8, :8], v[:8, -8:], v[-8:, :8], v[-8:, -8:]]).reshape(-1, 3), 0)
        sil = np.abs(v - bg).max(2) > 12
        iou = (sil & m_out).sum() / (sil | m_out).sum()
        print(f"silhouette IoU(view, aligned GT mask) = {iou:.3f}")


if __name__ == "__main__":
    main()
