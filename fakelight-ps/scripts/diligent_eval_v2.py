"""엔진 v2 채점: 커널 출력(out/{OBJ}_{IMG}, diligent_v2_template.py)의 재료를 합쳐 정답과 비교한다.

재료: SDM-UniPS 노멀 (ic_rot, ic_plain 32장, nlr_all), 같은 사진 RoSE, 단안 추정기 (DSINE, MoGe-2).
단안 추정기는 모델마다 좌표축 부호가 달라서, 정답이 아니라 우리 SDM 평균과의 채널별 부호 일치로 정한다.
합체 방식은 미리 정해둔 셋만 비교한다 (정답을 보고 고르지 않음):
  v1 레시피   : 저주파 = blur(avg(ic, nlr)), 고주파 = ic - blur(ic)       (σ=4, 버니에서 고정)
  전체 평균   : 모든 재료를 똑같이 평균
  역할 분담   : 저주파 = blur(전체 평균), 고주파 = ic - blur(ic)
단계별 전체 오차와 함께 저주파 오차(σ=8 흐림 후), 어두운 영역(입력 밝기 < 0.15) 오차를 기록한다.

사용법:
    python scripts/diligent_eval_v2.py results/diligent_v2/harvestPNG_025 --data data/DiLiGenT/pmsData
"""
import argparse
import json
import os

import numpy as np
import scipy.io as sio
from PIL import Image

from detail_metrics import masked_blur
from diligent_eval import SIGMA, norm, resize_normal, uncrop
from ps_eval import load_normal_png, mean_angular_error


def orient(n, ref, m):
    """단안 추정기 출력의 축 부호를 우리 좌표계에 맞춘다 (ref = 우리 SDM 평균, 정답 아님)."""
    s = np.sign((n[m] * ref[m]).sum(0))
    s[s == 0] = 1
    return norm(n * s)


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
    raw_path = os.path.join(args.folder, "input_crop_raw.png")
    inp = np.asarray(Image.open(raw_path if os.path.exists(raw_path) else os.path.join(args.folder, "input_crop.png"))
                     .convert("L"), dtype=np.float64) / 255
    # 원래 좌표 밝기 (uncrop은 벡터를 정규화하므로 스칼라는 따로 되돌린다)
    small = np.asarray(Image.fromarray(inp.astype(np.float32), "F").resize((crop["size"],) * 2, Image.BILINEAR))
    dark = np.ones(gmask.shape)
    y0, x0, s = crop["y0"], crop["x0"], crop["size"]
    ys, xs, ye, xe = max(0, y0), max(0, x0), min(gmask.shape[0], y0 + s), min(gmask.shape[1], x0 + s)
    dark[ys:ye, xs:xe] = small[ys - y0:ye - y0, xs - x0:xe - x0]

    sdm = lambda s: load_normal_png(os.path.join(args.folder, "sdm_results", f"{s}.data", "normal.png"), (F, F))
    src = {"IC-Light 회전 트릭": sdm("ic_rot"), "IC-Light 회전 없음": sdm("ic_plain"), "Neural LightRig": sdm("nlr_all")}
    ic = norm(src["IC-Light 회전 트릭"] + src["IC-Light 회전 없음"])
    ref = norm(ic + src["Neural LightRig"])
    rose_path = os.path.join(args.folder, "rose_normal.npy")
    if os.path.exists(rose_path):
        src["RoSE (같은 사진)"] = resize_normal(np.load(rose_path), F)
    for name, key in [("DSINE", "dsine"), ("MoGe-2", "moge2")]:
        p = os.path.join(args.folder, f"mono_{key}.npy")
        if os.path.exists(p):
            src[name] = orient(resize_normal(np.nan_to_num(np.load(p)), F), ref, cm)

    hf = lambda base: norm(masked_blur(base, cm, SIGMA) + (ic - masked_blur(ic, cm, SIGMA)))
    allavg = norm(sum(src.values()))
    stages = dict(src)
    stages["합체: v1 레시피"] = hf(ref)
    stages["합체: 전체 평균"] = allavg
    stages["합체: 역할 분담 (최종)"] = hf(allavg)

    # 원래 좌표 정답의 저주파 (σ=8)
    rows = {}
    for name, n in stages.items():
        full = uncrop(n * cm[..., None], crop, gmask.shape)
        mae, err = mean_angular_error(full, gt, gmask)
        lf = mean_angular_error(masked_blur(full, gmask, 8), masked_blur(gt, gmask, 8), gmask)[0]
        d = gmask & (dark < 0.15)
        rows[name] = (mae, lf, err[d].mean() if d.any() else float("nan"))

    final = uncrop(stages["합체: 역할 분담 (최종)"] * cm[..., None], crop, gmask.shape)
    Image.fromarray(((final * 0.5 + 0.5) * 255 * gmask[..., None]).astype(np.uint8)).save(
        os.path.join(args.folder, "final_normal.png"))
    stats = json.load(open(os.path.join(args.folder, "stats.json"))) if os.path.exists(
        os.path.join(args.folder, "stats.json")) else {}
    lines = [f"# {crop['obj']} (사진 {crop['img']}.png, 엔진 v2, 노출 게인 {stats.get('exposure_gain', '?')})", "",
             "| 단계 | 평균 각도 오차 | 저주파 오차 (σ=8) | 어두운 영역 오차 |", "|---|---|---|---|"]
    lines += [f"| {k} | **{v[0]:.2f}°** | {v[1]:.2f}° | {v[2]:.2f}° |" for k, v in rows.items()]
    text = "\n".join(lines)
    print(text)
    open(os.path.join(args.folder, "RESULT.md"), "w").write(text + "\n")
    json.dump({k: v[0] for k, v in rows.items()}, open(os.path.join(args.folder, "scores.json"), "w"), indent=2,
              ensure_ascii=False)


if __name__ == "__main__":
    main()
