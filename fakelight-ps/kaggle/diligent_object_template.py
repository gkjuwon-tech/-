# DiLiGenT 원정: 물체 하나를 사진 1장으로 처리한다 (버니에서 고정한 레시피, GPU 단계만).
#   입력: pmsData/{OBJ}/{IMG}.png 1장 + 물체 마스크 (단안 추정 벤치마크 관례대로 마스크는 사용)
#   1) 마스크 bbox로 정사각형 크롭 → 768px
#   2) IC-Light 16방향 × (회전 트릭 1세트 + 회전 없음 1세트), 시드 1
#   3) Neural LightRig 9방향 × 시드 3개
#   4) SDM-UniPS (내부 1024px)로 세트별로 풀기: ic_rot, ic_plain, nlr_all
# 합체, 디테일 보정, 원래 좌표로 되돌리기, 채점은 로컬에서 한다 (scripts/diligent_eval.py).
# 버니 레시피와의 차이: 시간 때문에 IC-Light 회전 없는 세트를 시드 3개(48장) 대신 시드 1개(16장)로 줄였다.
import gc, glob, json, os, shutil, subprocess, sys, time

def sh(cmd, check=True):
    print(f"\n$ {cmd}", flush=True)
    t = time.time()
    r = subprocess.run(cmd, shell=True)
    print(f"[exit={r.returncode} {time.time()-t:.0f}s]", flush=True)
    if check and r.returncode != 0:
        sys.exit(r.returncode)
    return r.returncode == 0

OBJ, IMG = "__OBJ__", "__IMG__"
W = "/kaggle/working"
OUT = f"{W}/out/{OBJ}_{IMG}"
os.makedirs(f"{OUT}/sets", exist_ok=True)
os.environ["HF_TOKEN"] = "__HF_TOKEN__"
T0 = time.time()
stats = {}

sh("pip install -q 'diffusers==0.31.0' 'transformers==4.46.3' 'huggingface_hub==0.36.2' 'peft==0.14.0' "
   "accelerate einops safetensors omegaconf pytorch-lightning megfile wandb")
sh(f"git clone -q https://github.com/lllyasviel/IC-Light.git {W}/IC-Light")
sh(f"git clone -q --depth 1 https://github.com/ZexinHe/Neural-LightRig.git {W}/Neural-LightRig")
sh(f"git clone -q https://github.com/satoshi-ikehata/SDM-UniPS-CVPR2023.git {W}/SDM-UniPS")
sh(f"curl -sSL -o {W}/sdm_ckpt.zip 'https://www.dropbox.com/s/yu8h6g0zp07mumd/checkpoint.zip?dl=1' "
   f"&& cd {W} && unzip -q -o sdm_ckpt.zip -d sdm_ckpt")
SDM_CKPT = os.path.dirname(os.path.dirname(glob.glob(f"{W}/sdm_ckpt/**/normal/*.pytmodel", recursive=True)[0]))
for f in glob.glob(f"{W}/Neural-LightRig/mld/**/*.py", recursive=True) + glob.glob(f"{W}/Neural-LightRig/mld/configs/*.yaml"):
    s = open(f).read()
    s2 = s.replace("stabilityai/stable-diffusion-2-1-unclip", "sd2-community/stable-diffusion-2-1-unclip") \
          .replace('"stabilityai/stable-diffusion-2-1"', '"sd2-community/stable-diffusion-2-1"')
    if s2 != s:
        open(f, "w").write(s2)

import cv2
import numpy as np
import torch
from PIL import Image

# ---------- 1) 입력 크롭 ----------
obj_dir = glob.glob(f"/kaggle/input/**/pmsData/{OBJ}", recursive=True)[0]
img = np.asarray(Image.open(f"{obj_dir}/{IMG}.png").convert("RGB"))
mask = np.asarray(Image.open(f"{obj_dir}/mask.png").convert("L")) > 127
ys, xs = np.nonzero(mask)
cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
half = int(max(ys.max() - ys.min(), xs.max() - xs.min()) * 0.55) + 1   # 여백 10%
F = 768
pad = half + 2
img_p = np.pad(img, ((pad, pad), (pad, pad), (0, 0)))
mask_p = np.pad(mask, ((pad, pad), (pad, pad)))
y0, x0 = int(round(cy)) - half + pad, int(round(cx)) - half + pad
crop = img_p[y0:y0 + 2 * half, x0:x0 + 2 * half]
cmask = mask_p[y0:y0 + 2 * half, x0:x0 + 2 * half]
crop = cv2.resize(crop, (F, F), interpolation=cv2.INTER_AREA)
cmask = cv2.resize(cmask.astype(np.uint8) * 255, (F, F), interpolation=cv2.INTER_NEAREST) > 127
# 원래 좌표로 되돌리기 위한 정보: 크롭 원점(원본 좌표) = (y0-pad, x0-pad), 크기 2*half
json.dump(dict(obj=OBJ, img=IMG, y0=int(y0 - pad), x0=int(x0 - pad), size=int(2 * half), frame=F,
               orig_shape=list(mask.shape)), open(f"{OUT}/crop.json", "w"), indent=2)
Image.fromarray(crop).save(f"{OUT}/input_crop.png")
Image.fromarray((cmask * 255).astype(np.uint8)).save(f"{OUT}/mask_crop.png")
fg_gray = np.where(cmask[..., None], crop, 127).astype(np.uint8)      # 배경 회색 (IC-Light 관례)

# ---------- 2) IC-Light ----------
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
ANGLES = [i * 22.5 for i in range(16)]


def gradient_bg(angle_deg, size):
    a = np.deg2rad(angle_deg)
    dx, dy = np.cos(a), np.sin(a)
    xs_ = np.linspace(-1, 1, size)
    xn, yn = np.meshgrid(xs_, -xs_)
    proj = (xn * dx + yn * dy) / (abs(dx) + abs(dy))
    v = (127.5 + 127.5 * proj).clip(0, 255).astype(np.uint8)
    return np.stack([v] * 3, -1)


def plan_rotation(t):
    dist = lambda a: min(abs(a % 360), abs(a % 360 - 180), abs(a % 360 - 360))
    k = min(range(4), key=lambda k: dist(t + 90 * k))
    return k, (t + 90 * k) % 360


prompt = "an object, photograph"
for set_name, use_rot in [("ic_rot", True), ("ic_plain", False)]:
    t0 = time.time()
    for i, ang in enumerate(ANGLES):
        k, d = plan_rotation(ang) if use_rot else (0, ang)
        fg_r = np.ascontiguousarray(np.rot90(fg_gray, k))
        pixels, _ = ic["process"](fg_r, gradient_bg(d, F), prompt, 512, 512, 1, 1, 25, "best quality",
                                  "lowres, bad anatomy, bad hands, cropped, worst quality", 2.0, 1.5, 0.5,
                                  ic["BGSource"].UPLOAD.value)
        out = np.ascontiguousarray(np.rot90((pixels[0] * 255).clip(0, 255).astype(np.uint8), -k))
        Image.fromarray(out).resize((F, F), Image.BICUBIC).save(f"{OUT}/sets/{set_name}__L_{i:02d}.png")
    Image.fromarray((cmask * 255).astype(np.uint8)).save(f"{OUT}/sets/{set_name}__mask.png")
    stats[set_name] = round(time.time() - t0, 1)
    print(set_name, stats[set_name], flush=True)
ic.clear(); gc.collect(); torch.cuda.empty_cache()

# ---------- 3) Neural LightRig ----------
os.chdir(f"{W}/Neural-LightRig")
sys.path.insert(0, f"{W}/Neural-LightRig")
from huggingface_hub import snapshot_download
from inference import prepare_stage1_mld
from utils.vis import replace_bg_preserving_alpha
ckpt = snapshot_download(repo_id="zxhezexin/neural-lightrig-mld-and-recon", local_dir=f"{W}/nlr_ckpt",
                         allow_patterns=["mld.pt", "config.json"])
model1 = prepare_stage1_mld(cfg_path="./mld/configs/infer.yaml", ckpt_path=os.path.join(ckpt, "mld.pt"))
rgba = Image.fromarray(np.dstack([crop, (cmask * 255).astype(np.uint8)])).resize((512, 512))
t0 = time.time()
k_img = 0
for seed in [511, 1, 2]:
    torch.manual_seed(seed); np.random.seed(seed)
    grid = model1.pipeline(image=replace_bg_preserving_alpha(rgba, 255).convert("RGB"), guidance_scale=2.0,
                           guidance_rescale=0.7, num_inference_steps=75).images[0]
    g = np.asarray(grid.convert("RGB"))
    T = g.shape[0] // 3
    for i in range(9):
        tile = g[(i // 3) * T:(i // 3 + 1) * T, (i % 3) * T:(i % 3 + 1) * T]
        Image.fromarray(tile).resize((F, F), Image.BICUBIC).save(f"{OUT}/sets/nlr_all__L_{k_img:02d}.png")
        k_img += 1
Image.fromarray((cmask * 255).astype(np.uint8)).save(f"{OUT}/sets/nlr_all__mask.png")
stats["nlr"] = round(time.time() - t0, 1)
del model1; gc.collect(); torch.cuda.empty_cache()

# ---------- 4) SDM-UniPS ----------
os.chdir(f"{W}/SDM-UniPS")
for set_name in ["ic_rot", "ic_plain", "nlr_all"]:
    root = f"{W}/sdm_sets/{set_name}"
    d = f"{root}/{set_name}.data"
    os.makedirs(d, exist_ok=True)
    paths = sorted(glob.glob(f"{OUT}/sets/{set_name}__L_*.png"))
    for i, p in enumerate(paths):
        im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        cv2.imwrite(f"{d}/L_{i:03d}.png", cv2.resize(im, (2 * F, 2 * F), interpolation=cv2.INTER_CUBIC))
    cv2.imwrite(f"{d}/mask.png", cv2.resize((cmask * 255).astype(np.uint8), (2 * F, 2 * F), interpolation=cv2.INTER_NEAREST))
    for k in [len(paths), 18, 16, 9]:
        if k > len(paths):
            continue
        if sh(f"PYTORCH_ALLOC_CONF=expandable_segments:True python sdm_unips/main.py --session_name sdm_{set_name} "
              f"--test_dir {root} --checkpoint {SDM_CKPT} --target normal --max_image_num {k} --scalable", check=False):
            shutil.copytree(f"sdm_{set_name}/results/{set_name}.data", f"{OUT}/sdm_results/{set_name}.data")
            stats[f"sdm_{set_name}"] = k
            break
    shutil.rmtree(root, ignore_errors=True)

# ---------- 5) 같은 사진으로 RoSE (경쟁자, 공식 test.py와 같은 최소제곱) ----------
try:
    sh(f"git clone -q --depth 1 https://github.com/LMozart/ICLR2026-RoSE.git {W}/RoSE")
    sys.path.insert(0, f"{W}/RoSE")
    import torch as th
    from einops import rearrange, repeat
    from torchvision import transforms
    from test import load_primary_models
    rckpt = snapshot_download("Xinhua694/RoSE", local_dir=f"{W}/rose_ckpt")
    pipe, unet = load_primary_models("chenguolin/sv3d-diffusers", os.path.join(rckpt, "ckpt"))
    pipe = pipe.to("cuda", dtype=th.float16); unet.to("cuda", dtype=th.float16)
    gray = np.repeat(crop.mean(2, keepdims=True), 3, 2).astype(np.uint8)
    g = th.Generator(device="cuda"); g.manual_seed(42)
    with th.no_grad():
        frames = pipe(image=Image.fromarray(gray).resize((576, 576), Image.NEAREST), width=576, height=576,
                      num_frames=9, num_inference_steps=25, decode_chunk_size=8, polars_rad=[np.deg2rad(90)] * 9,
                      azimuths_rad=[0] * 9, output_type="pil", generator=g).frames[0]
    i_ = th.arange(9, dtype=th.float32)
    az, el = ((i_ / 9) * 360 % 360).deg2rad(), th.tensor([45.0] * 9).deg2rad()
    lgt = th.stack([th.cos(el) * th.sin(az), th.cos(el) * th.cos(az), th.sin(el)], 1).cuda()
    v = th.stack([transforms.ToTensor()(f) for f in frames]).cuda().float()
    sh_ = rearrange(v.mean(1), "b h w -> (h w) b 1").clip(0); sh_ = sh_ / sh_.max()
    msk = (sh_ > 0).float(); L = repeat(lgt, "f c -> p f c", p=sh_.shape[0])
    x = th.linalg.solve(L.transpose(1, 2) @ (msk * L) + 1e-6 * th.eye(3, device="cuda"), L.transpose(1, 2) @ (msk * sh_))
    np.save(f"{OUT}/rose_normal.npy", th.nn.functional.normalize(x, dim=1)[..., 0].reshape(576, 576, 3).cpu().numpy())
    stats["rose"] = "ok"
except Exception as e:
    import traceback; traceback.print_exc()
    stats["rose"] = repr(e)

stats["total_seconds"] = round(time.time() - T0, 1)
json.dump(stats, open(f"{OUT}/stats.json", "w"), indent=2)
print(json.dumps(stats, indent=2), flush=True)
