# 가짜 조명 → 광도 스테레오 (v2): IC-Light(fbc)로 조명 방향만 바꾼 16방향 × 시드 3세트를 만들고 SDM-UniPS로 노멀을 푼다.
# 대조군으로 진짜 조명 렌더(K장)도 같은 SDM-UniPS에 넣는다. 채점은 로컬(scripts/ps_eval.py)에서 한다.
# 입력은 Kaggle 데이터셋으로 마운트한다 (build_kernel.py --dataset).
import gc, glob, json, os, shutil, subprocess, sys, time

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
IN = f"{W}/inputs"
os.makedirs(OUT, exist_ok=True)

# 입력은 비공개 데이터셋(flyjw12/fakelight-inputs)으로 마운트된다. 스크립트에 base64로 넣으면
# Kaggle 크기 제한(약 1MB)에 걸린다. "real__X" 파일은 real_lights.data/X로 풀어 SDM-UniPS 형식을 만든다.
src_dir = next(d for d in glob.glob("/kaggle/input/**/", recursive=True)
               if os.path.exists(os.path.join(d, "jugg_front.png")))
os.makedirs(f"{IN}/real_lights.data", exist_ok=True)
for f in os.listdir(src_dir):
    dst = f"{IN}/real_lights.data/{f[len('real__'):]}" if f.startswith("real__") else f"{IN}/{f}"
    shutil.copy(os.path.join(src_dir, f), dst)
print("inputs:", sorted(glob.glob(f"{IN}/**/*", recursive=True)), flush=True)

sh("pip install -q 'diffusers==0.32.2' 'transformers==4.49.0' 'huggingface_hub==0.36.2' 'peft==0.14.0' "
   "accelerate einops safetensors")
sh(f"git clone -q https://github.com/lllyasviel/IC-Light.git {W}/IC-Light")
sh(f"git clone -q https://github.com/satoshi-ikehata/SDM-UniPS-CVPR2023.git {W}/SDM-UniPS")
sh(f"curl -sSL -o {W}/sdm_ckpt.zip 'https://www.dropbox.com/s/yu8h6g0zp07mumd/checkpoint.zip?dl=1' "
   f"&& cd {W} && unzip -q -o sdm_ckpt.zip -d sdm_ckpt")
nml_model = glob.glob(f"{W}/sdm_ckpt/**/normal/*.pytmodel", recursive=True)
assert nml_model, "SDM-UniPS normal checkpoint not found"
SDM_CKPT = os.path.dirname(os.path.dirname(nml_model[0]))
print("SDM-UniPS checkpoint:", SDM_CKPT, flush=True)

import numpy as np
import torch
from PIL import Image

CFG = dict(
    prompt="a white clay sculpture of a bunny",
    a_prompt="best quality",
    n_prompt="lowres, bad anatomy, bad hands, cropped, worst quality",
    seeds=[12345, 1, 2],  # 시드별로 한 세트씩. 세트 안에서는 모든 방향에 같은 시드 → 방향 간 일관성
    steps=25,
    cfg=2.0,
    low_res=512,          # IC-Light 1단계 해상도, 2단계에서 ×1.5 = 768
    highres_scale=1.5,
    highres_denoise=0.5,
    num_dirs=16,          # 화면 기준 0°(오른쪽)부터 반시계 22.5° 간격
    bg_range=(0, 255),    # v1은 (32, 224). 아래쪽 조명이 약하게 나와서 대비를 최대로
    sdm_max_images=64,
    upscale_variant=2,    # SDM-UniPS는 크롭을 512 배수로 내림 → 2배로 키워 넣으면 내부 1024px
)
print(json.dumps(CFG, indent=2), torch.cuda.get_device_name(0), flush=True)
ANGLES = [i * 360 / CFG["num_dirs"] for i in range(CFG["num_dirs"])]

# ---------- IC-Light (fbc): 데모 파일에서 UI만 빼고 모델/함수를 불러온다 ----------
os.chdir(f"{W}/IC-Light")
sys.path.insert(0, f"{W}/IC-Light")  # briarmbg.py import용
src = open("gradio_demo_bg.py").read()
src = src[:src.index("block = gr.Blocks()")]
for old, new in [
    ("import gradio as gr\n", ""),
    ("import db_examples\n", ""),
    # T4는 bf16 연산을 제대로 지원하지 않아서 VAE는 fp32로 (작아서 메모리 부담 없음)
    ("vae = vae.to(device=device, dtype=torch.bfloat16)", "vae = vae.to(device=device, dtype=torch.float32)"),
]:
    assert src.count(old) == 1, old
    src = src.replace(old, new)
ic = {"__name__": "iclight_headless"}
exec(compile(src, "gradio_demo_bg.py", "exec"), ic)


def gradient_bg(angle_deg, size):
    """빛이 오는 쪽이 밝은 선형 그라데이션 배경 (IC-Light 데모의 Left/Right/Top/Bottom 일반화)."""
    a = np.deg2rad(angle_deg)
    dx, dy = np.cos(a), np.sin(a)
    xs = np.linspace(-1, 1, size)
    xn, yn = np.meshgrid(xs, -xs)  # yn: 위쪽이 +
    proj = (xn * dx + yn * dy) / (abs(dx) + abs(dy))
    lo, hi = CFG["bg_range"]
    v = ((lo + hi) / 2 + (hi - lo) / 2 * proj).clip(0, 255).astype(np.uint8)
    return np.stack([v] * 3, -1)


def relight_subject(name, fg_rgb, seed):
    out = f"{OUT}/{name}_s{seed}.data"
    os.makedirs(out, exist_ok=True)
    input_fg, matting = ic["run_rmbg"](fg_rgb, sigma=16)
    t0 = time.time()
    for i, ang in enumerate(ANGLES):
        bg = gradient_bg(ang, fg_rgb.shape[0])
        pixels, _ = ic["process"](
            input_fg, bg, CFG["prompt"], CFG["low_res"], CFG["low_res"], 1, seed, CFG["steps"],
            CFG["a_prompt"], CFG["n_prompt"], CFG["cfg"], CFG["highres_scale"], CFG["highres_denoise"],
            ic["BGSource"].UPLOAD.value)
        Image.fromarray((pixels[0] * 255).clip(0, 255).astype(np.uint8)).save(f"{out}/L_{i:02d}.png")
    size = Image.open(f"{out}/L_00.png").size
    m = Image.fromarray((matting[..., 0] * 255).clip(0, 255).astype(np.uint8)).resize(size, Image.BILINEAR)
    Image.fromarray(((np.asarray(m) > 127) * 255).astype(np.uint8)).save(f"{out}/mask.png")
    json.dump(dict(CFG, seed=seed, angles=ANGLES, seconds=round(time.time() - t0, 1)),
              open(f"{out}/relight.json", "w"), indent=2)
    print(f"  {name} seed {seed}: {len(ANGLES)} dirs in {time.time()-t0:.0f}s", flush=True)


subjects = {
    # 메인: MV-Adapter + Juggernaut가 만든 정면 뷰 (회색 배경 RGB)
    "jugg": np.asarray(Image.open(f"{IN}/jugg_front.png").convert("RGB")),
}
# 대조군: 원본 렌더 (생성 단계 오차 없이 리라이팅 오차만 보기 위함)
rgba = np.asarray(Image.open(f"{IN}/render_front_rgba.png")).astype(np.float32) / 255.0
subjects["render"] = ((rgba[..., :3] * rgba[..., 3:] + 0.5 * (1 - rgba[..., 3:])) * 255).astype(np.uint8)

torch.cuda.reset_peak_memory_stats()
for name, img in subjects.items():
    for seed in CFG["seeds"]:
        relight_subject(name, img, seed)
print("IC-Light peak mem GB:", round(torch.cuda.max_memory_allocated() / 1e9, 2), flush=True)
ic.clear(); gc.collect(); torch.cuda.empty_cache()

# ---------- SDM-UniPS 입력 세트 ----------
#   {name}_s{seed}   : 시드 하나, 16방향 (로컬에서 시드별 노멀을 평균내 앙상블도 채점)
#   {name}_all       : 모든 시드를 한 번에 (48장)
#   {name}_all_x2    : 위와 같지만 2배 업스케일 → SDM-UniPS 내부 1024px
TEST = f"{W}/sdm_test"
os.makedirs(TEST, exist_ok=True)


def add_set(dst, srcs, scale=1):
    os.makedirs(dst, exist_ok=True)
    k = 0
    for s in srcs:
        for p in sorted(glob.glob(f"{s}/L_*.png")):
            im = Image.open(p)
            if scale != 1:
                im = im.resize((im.width * scale, im.height * scale), Image.BICUBIC)
            im.save(f"{dst}/L_{k:03d}.png"); k += 1
    m = Image.open(f"{srcs[0]}/mask.png")
    if scale != 1:
        m = m.resize((m.width * scale, m.height * scale), Image.NEAREST)
    m.save(f"{dst}/mask.png")


for name in subjects:
    seed_dirs = [f"{OUT}/{name}_s{s}.data" for s in CFG["seeds"]]
    for sd in seed_dirs:
        add_set(f"{TEST}/{os.path.basename(sd)}", [sd])
    add_set(f"{TEST}/{name}_all.data", seed_dirs)
    add_set(f"{TEST}/{name}_all_x2.data", seed_dirs, scale=CFG["upscale_variant"])
shutil.copytree(f"{IN}/real_lights.data", f"{TEST}/real_lights.data")
for f in glob.glob(f"{TEST}/real_lights.data/*"):
    if not (os.path.basename(f).startswith("L_") or f.endswith("mask.png")):
        os.remove(f)

os.chdir(f"{W}/SDM-UniPS")
t0 = time.time()
sh(f"python sdm_unips/main.py --session_name sdm_out --test_dir {TEST} --checkpoint {SDM_CKPT} "
   f"--target normal_and_brdf --max_image_num {CFG['sdm_max_images']} --scalable")
print("SDM-UniPS seconds:", round(time.time() - t0, 1), flush=True)
shutil.copytree(f"{W}/SDM-UniPS/sdm_out/results", f"{OUT}/sdm_results")
print(sorted(glob.glob(f"{OUT}/sdm_results/*/normal.png")), flush=True)
