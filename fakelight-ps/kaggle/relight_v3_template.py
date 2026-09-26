# 가짜 조명 → 광도 스테레오 (v3): 병목(리라이팅) 진단 결과에 맞춘 세 가지 처방을 따로따로 측정한다.
#   ① 선형화: IC-Light 출력은 화면용 sRGB라서 선형 밝기로 바꿔 16비트로 넣는다 (SDM-UniPS는 선형 렌더로 학습)
#   ② 환경광 제거: 방향별 이미지의 픽셀별 최솟값(≈ 환경광)을 빼서 방향광 성분만 남긴다
#   ③ 회전 트릭: IC-Light는 좌우 조명을 가장 물리적으로 그린다 (진단 R² 0.77~0.85). 입력을 90° 단위로 돌려서
#      모든 요청 방향을 좌우 조명으로 바꿔 그리고 결과를 되돌린다. 90° 회전은 픽셀이 정확히 보존된다.
# A: v2 리라이팅(데이터셋 flyjw12/fakelight-relights-v2)에 ①② 적용, B: 회전 트릭으로 새로 리라이팅 + ①②.
# 모든 세트는 v2에서 가장 좋았던 2배 업스케일(SDM-UniPS 내부 1024px)로 푼다.
import gc, glob, json, os, re, shutil, subprocess, sys, time

def sh(cmd, check=True):
    print(f"\n$ {cmd}", flush=True)
    t = time.time()
    r = subprocess.run(cmd, shell=True)
    print(f"[exit={r.returncode} {time.time()-t:.0f}s]", flush=True)
    if check and r.returncode != 0:
        sys.exit(r.returncode)
    return r.returncode == 0

W = "/kaggle/working"
OUT = f"{W}/out"
os.makedirs(OUT, exist_ok=True)

sh("pip install -q 'diffusers==0.32.2' 'transformers==4.49.0' 'huggingface_hub==0.36.2' 'peft==0.14.0' "
   "accelerate einops safetensors")
sh(f"git clone -q https://github.com/lllyasviel/IC-Light.git {W}/IC-Light")
sh(f"git clone -q https://github.com/satoshi-ikehata/SDM-UniPS-CVPR2023.git {W}/SDM-UniPS")
sh(f"curl -sSL -o {W}/sdm_ckpt.zip 'https://www.dropbox.com/s/yu8h6g0zp07mumd/checkpoint.zip?dl=1' "
   f"&& cd {W} && unzip -q -o sdm_ckpt.zip -d sdm_ckpt")
SDM_CKPT = os.path.dirname(os.path.dirname(glob.glob(f"{W}/sdm_ckpt/**/normal/*.pytmodel", recursive=True)[0]))

import cv2
import numpy as np
import torch
from PIL import Image

inputs_dir = next(d for d in glob.glob("/kaggle/input/**/", recursive=True) if os.path.exists(f"{d}/jugg_front.png"))
relit_dir = next(d for d in glob.glob("/kaggle/input/**/", recursive=True) if glob.glob(f"{d}/jugg_s*__L_00.png"))

CFG = dict(
    prompt="a white clay sculpture of a bunny",
    a_prompt="best quality",
    n_prompt="lowres, bad anatomy, bad hands, cropped, worst quality",
    rot_seeds=[1, 2],
    steps=25, cfg=2.0, low_res=512, highres_scale=1.5, highres_denoise=0.5,
    num_dirs=16,
    bg_range=(0, 255),
    upscale=2,
)
ANGLES = [i * 360 / CFG["num_dirs"] for i in range(CFG["num_dirs"])]
print(json.dumps(CFG, indent=2), flush=True)

# ---------- B: 회전 트릭 리라이팅 ----------
os.chdir(f"{W}/IC-Light")
sys.path.insert(0, f"{W}/IC-Light")
src = open("gradio_demo_bg.py").read()
src = src[:src.index("block = gr.Blocks()")]
for old, new in [("import gradio as gr\n", ""), ("import db_examples\n", ""),
                 ("vae = vae.to(device=device, dtype=torch.bfloat16)", "vae = vae.to(device=device, dtype=torch.float32)")]:
    assert src.count(old) == 1, old
    src = src.replace(old, new)
ic = {"__name__": "iclight_headless"}
exec(compile(src, "gradio_demo_bg.py", "exec"), ic)


def gradient_bg(angle_deg, size):
    a = np.deg2rad(angle_deg)
    dx, dy = np.cos(a), np.sin(a)
    xs = np.linspace(-1, 1, size)
    xn, yn = np.meshgrid(xs, -xs)
    proj = (xn * dx + yn * dy) / (abs(dx) + abs(dy))
    lo, hi = CFG["bg_range"]
    v = ((lo + hi) / 2 + (hi - lo) / 2 * proj).clip(0, 255).astype(np.uint8)
    return np.stack([v] * 3, -1)


def plan_rotation(target):
    """target 방향을 좌(180°)/우(0°) 조명으로 바꾸는 90° 배수 회전 k를 고른다.
    이미지를 반시계로 90k° 돌리면 원래 t방향 빛은 돌린 이미지에서 t+90k 방향이 된다."""
    def dist(a):
        a %= 360
        return min(abs(a), abs(a - 180), abs(a - 360))
    k = min(range(4), key=lambda k: dist(target + 90 * k))
    return k, (target + 90 * k) % 360


fg = np.asarray(Image.open(f"{inputs_dir}/jugg_front.png").convert("RGB"))
input_fg, matting = ic["run_rmbg"](fg, sigma=16)
rot_plan = {a: plan_rotation(a) for a in ANGLES}
print("rotation plan (target -> k, rendered dir):", rot_plan, flush=True)
for seed in CFG["rot_seeds"]:
    out = f"{OUT}/jugg_rot_s{seed}.data"
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    for i, ang in enumerate(ANGLES):
        k, d = rot_plan[ang]
        fg_r = np.ascontiguousarray(np.rot90(input_fg, k))
        pixels, _ = ic["process"](
            fg_r, gradient_bg(d, fg_r.shape[0]), CFG["prompt"], CFG["low_res"], CFG["low_res"], 1, seed,
            CFG["steps"], CFG["a_prompt"], CFG["n_prompt"], CFG["cfg"], CFG["highres_scale"],
            CFG["highres_denoise"], ic["BGSource"].UPLOAD.value)
        img = np.ascontiguousarray(np.rot90((pixels[0] * 255).clip(0, 255).astype(np.uint8), -k))
        Image.fromarray(img).save(f"{out}/L_{i:02d}.png")
    size = Image.open(f"{out}/L_00.png").size
    m = Image.fromarray((matting[..., 0] * 255).clip(0, 255).astype(np.uint8)).resize(size, Image.BILINEAR)
    Image.fromarray(((np.asarray(m) > 127) * 255).astype(np.uint8)).save(f"{out}/mask.png")
    json.dump(dict(CFG, seed=seed, angles=ANGLES, rotation=rot_plan, seconds=round(time.time() - t0, 1)),
              open(f"{out}/relight.json", "w"), indent=2, default=str)
    print(f"rot seed {seed}: {time.time()-t0:.0f}s", flush=True)
ic.clear(); gc.collect(); torch.cuda.empty_cache()


# ---------- 전처리 변형 + SDM-UniPS ----------
def srgb2lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def load_group(paths):
    return np.stack([np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0 for p in paths])


def write_set(dst, imgs, mask, mode, scale):
    """imgs: (K,H,W,3) sRGB [0,1]. mode: srgb / lin / linamb"""
    os.makedirs(dst, exist_ok=True)
    x = imgs if mode == "srgb" else srgb2lin(imgs)
    if mode == "linamb":
        x = np.clip(x - x.min(0, keepdims=True), 0, None)
    x = x / (x.max() + 1e-8)
    for i, im in enumerate(x):
        if scale != 1:
            im = cv2.resize(im, (im.shape[1] * scale, im.shape[0] * scale), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(f"{dst}/L_{i:03d}.png", (np.clip(im, 0, 1) * 65535).astype(np.uint16)[..., ::-1])
    m = mask if scale == 1 else cv2.resize(mask, (mask.shape[1] * scale, mask.shape[0] * scale),
                                           interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(f"{dst}/mask.png", m)


groups = {
    "v2_s1": sorted(glob.glob(f"{relit_dir}/jugg_s1__L_*.png")),
    "v2_all": sorted(glob.glob(f"{relit_dir}/jugg_s*__L_*.png")),
    "rot_s1": sorted(glob.glob(f"{OUT}/jugg_rot_s1.data/L_*.png")),
    "rot_all": sorted(glob.glob(f"{OUT}/jugg_rot_s*.data/L_*.png")),
}
masks = {
    "v2": cv2.imread(f"{relit_dir}/jugg_s1__mask.png", cv2.IMREAD_GRAYSCALE),
    "rot": cv2.imread(f"{OUT}/jugg_rot_s1.data/mask.png", cv2.IMREAD_GRAYSCALE),
}
SETS = [(f"jugg_{g}_x2_{mode}", g, mode) for g in groups for mode in ("srgb", "lin", "linamb")]

os.chdir(f"{W}/SDM-UniPS")
env = "PYTORCH_ALLOC_CONF=expandable_segments:True"
summary = {}
for name, g, mode in SETS:
    root = f"{W}/sdm_sets/{name}"
    paths = groups[g]
    write_set(f"{root}/{name}.data", load_group(paths), masks[g.split("_")[0]], mode, CFG["upscale"])
    ok = False
    for k in [len(paths), 32, 24, 16]:
        if k > len(paths):
            continue
        t0 = time.time()
        ok = sh(f"{env} python sdm_unips/main.py --session_name sdm_{name} --test_dir {root} "
                f"--checkpoint {SDM_CKPT} --target normal --max_image_num {k} --scalable", check=False)
        if ok:
            summary[name] = dict(images=k, of=len(paths), seconds=round(time.time() - t0, 1))
            shutil.copytree(f"sdm_{name}/results/{name}.data", f"{OUT}/sdm_results/{name}.data")
            break
    if not ok:
        summary[name] = dict(error="all attempts failed", of=len(paths))
    shutil.rmtree(root, ignore_errors=True)
    print(name, summary[name], flush=True)

json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2), flush=True)
