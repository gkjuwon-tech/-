"""DiLiGenT 물체 하나의 비교 그림: 입력 / 정답 / 짭광 최종 / RoSE, 아래 줄은 오차 지도."""
import json
import os
import sys

import numpy as np
import scipy.io as sio
from PIL import Image, ImageDraw, ImageFont

from diligent_eval import resize_normal, uncrop
from ps_eval import mean_angular_error, load_normal_png


def main(folder, data="data/DiLiGenT/pmsData"):
    c = json.load(open(os.path.join(folder, "crop.json")))
    gt = sio.loadmat(os.path.join(data, c["obj"], "Normal_gt.mat"))["Normal_gt"]
    gm = np.asarray(Image.open(os.path.join(data, c["obj"], "mask.png")).convert("L")) > 127
    cm = np.asarray(Image.open(os.path.join(folder, "mask_crop.png")).convert("L")) > 127
    ours = load_normal_png(os.path.join(folder, "final_normal.png"), gm.shape)
    rose = uncrop(resize_normal(np.load(os.path.join(folder, "rose_normal.npy")), c["frame"]) * cm[..., None], c, gm.shape)
    ys, xs = np.nonzero(gm)
    y0, y1, x0, x1 = max(ys.min() - 10, 0), ys.max() + 10, max(xs.min() - 10, 0), xs.max() + 10
    T = 300
    font = ImageFont.load_default(size=20)

    def tile(a, label):
        im = Image.fromarray(a[y0:y1, x0:x1]).convert("RGB")
        im = im.resize((T, int(T * im.height / im.width)))
        d = ImageDraw.Draw(im); d.rectangle([0, 0, T, 28], fill="white"); d.text((5, 4), label, fill="black", font=font)
        return im

    nimg = lambda n: ((n * 0.5 + 0.5).clip(0, 1) * 255 * gm[..., None]).astype(np.uint8)
    def emap(n):
        e = np.clip(mean_angular_error(n, gt, gm)[1] / 60, 0, 1)
        return (np.stack([e, e * 0.3, 1 - e], 2) * 255 * gm[..., None]).astype(np.uint8)
    inp = np.zeros(gm.shape + (3,), np.uint8)
    ic = Image.open(os.path.join(folder, "input_crop.png")).resize((c["size"], c["size"]))
    full = np.zeros(gm.shape + (3,), np.uint8)
    a = np.asarray(ic)
    yy0, xx0 = c["y0"], c["x0"]
    ys_, xs_ = max(0, yy0), max(0, xx0)
    ye_, xe_ = min(gm.shape[0], yy0 + c["size"]), min(gm.shape[1], xx0 + c["size"])
    full[ys_:ye_, xs_:xe_] = a[ys_ - yy0:ye_ - yy0, xs_ - xx0:xe_ - xx0]
    top = [tile(full, "input"), tile(nimg(gt), "GT"),
           tile(nimg(ours), f"ours {mean_angular_error(ours, gt, gm)[0]:.1f}deg"),
           tile(nimg(rose), f"RoSE {mean_angular_error(rose, gt, gm)[0]:.1f}deg")]
    bot = [Image.new("RGB", top[0].size, "white"), Image.new("RGB", top[0].size, "white"),
           tile(emap(ours), "ours error (0-60)"), tile(emap(rose), "RoSE error (0-60)")]
    H = top[0].height
    canvas = Image.new("RGB", (4 * T, 2 * H), "white")
    for i, (t, b) in enumerate(zip(top, bot)):
        canvas.paste(t, (i * T, 0)); canvas.paste(b, (i * T, H))
    canvas.save(os.path.join(folder, "figure.png"))


if __name__ == "__main__":
    main(*sys.argv[1:])
