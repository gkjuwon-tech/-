# 경쟁자 채점: RoSE (ICLR 2026 Oral, DiLiGenT 단일 이미지 SOTA 16.36°)를 버니에 돌린다.
# RoSE = SV3D 기반 비디오 모델로 '링 조명 음영 9프레임'을 만들고 최소제곱으로 노멀을 푼다 (공식 test.py와 같은 과정).
# 공식 노멀과 함께 음영 프레임도 저장해서, 같은 음영을 SDM-UniPS로 풀었을 때도 비교한다.
import glob, json, os, shutil, subprocess, sys, time

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
os.makedirs(f"{OUT}/sets", exist_ok=True)
ROSE = f"{W}/RoSE"
sh("pip install -q 'diffusers==0.31.0' 'transformers==4.43.3' 'huggingface_hub==0.36.2' accelerate einops")
sh(f"git clone -q --depth 1 https://github.com/LMozart/ICLR2026-RoSE.git {ROSE}")
sh(f"git clone -q https://github.com/satoshi-ikehata/SDM-UniPS-CVPR2023.git {W}/SDM-UniPS")
sh(f"curl -sSL -o {W}/sdm_ckpt.zip 'https://www.dropbox.com/s/yu8h6g0zp07mumd/checkpoint.zip?dl=1' "
   f"&& cd {W} && unzip -q -o sdm_ckpt.zip -d sdm_ckpt")
SDM_CKPT = os.path.dirname(os.path.dirname(glob.glob(f"{W}/sdm_ckpt/**/normal/*.pytmodel", recursive=True)[0]))

import cv2
import numpy as np
import torch as th
from einops import rearrange, repeat
from PIL import Image
from torchvision import transforms
from huggingface_hub import snapshot_download

sys.path.insert(0, ROSE)
from test import load_primary_models

ckpt = snapshot_download("Xinhua694/RoSE", local_dir=f"{W}/rose_ckpt")
pipeline, unet = load_primary_models("chenguolin/sv3d-diffusers", os.path.join(ckpt, "ckpt"))
dev = th.device("cuda")
pipeline = pipeline.to(dev, dtype=th.float16)
unet.to(dev, dtype=th.float16)

FRAMES, SIZE, SEED = 9, 576, 42
inputs_dir = next(d for d in glob.glob("/kaggle/input/**/", recursive=True) if os.path.exists(f"{d}/jugg_front.png"))
relit_dir = next(d for d in glob.glob("/kaggle/input/**/", recursive=True) if os.path.exists(f"{d}/jugg_s1__mask.png"))
jugg = np.asarray(Image.open(f"{inputs_dir}/jugg_front.png").convert("RGB")).astype(np.float32)
jmask = np.asarray(Image.open(f"{relit_dir}/jugg_s1__mask.png").convert("L")) > 127   # 배경 제거 마스크 (정답 아님)
rgba = np.asarray(Image.open(f"{inputs_dir}/render_front_rgba.png").convert("RGBA")).astype(np.float32)
# RoSE는 흑백으로 바꿔 쓴다. 배경은 검정(음영 데이터 관례)과 원래 배경 두 가지로 돌린다
subjects = {
    "jugg_black": jugg * jmask[..., None],
    "jugg_orig": jugg,
    "render_black": rgba[..., :3] * (rgba[..., 3:] / 255.0),
}
masks = {"jugg": (jmask * 255).astype(np.uint8), "render": (rgba[..., 3] > 127).astype(np.uint8) * 255}

i = th.arange(FRAMES, dtype=th.float32)
az = ((i / FRAMES) * 360 % 360).deg2rad()
el = th.tensor([45.0] * FRAMES).deg2rad()
lgt = th.stack([th.cos(el) * th.sin(az), th.cos(el) * th.cos(az), th.sin(el)], 1)   # 공식 test.py와 동일
stats = {}
for name, img in subjects.items():
    gray = np.repeat(img.mean(2, keepdims=True), 3, 2).clip(0, 255).astype(np.uint8)
    inp = Image.fromarray(gray).resize((SIZE, SIZE), Image.NEAREST)
    g = th.Generator(device=dev); g.manual_seed(SEED)
    t0 = time.time()
    with th.no_grad():
        frames = pipeline(image=inp, width=SIZE, height=SIZE, num_frames=FRAMES, num_inference_steps=25,
                          decode_chunk_size=8, polars_rad=[np.deg2rad(90)] * FRAMES, azimuths_rad=[0] * FRAMES,
                          output_type="pil", generator=g).frames[0]
    # 공식 최소제곱 (test.py 4단계와 동일)
    v = th.stack([transforms.ToTensor()(f) for f in frames]).to(dev).float()
    shading = rearrange(v.mean(1), "b h w -> (h w) b 1").clip(0)
    shading = shading / shading.max()
    msk = (shading > 0).float()
    L = repeat(lgt, "f c -> p f c", p=shading.shape[0]).to(dev)
    M = L.transpose(1, 2) @ (msk * L) + 1e-6 * th.eye(3, device=dev)
    x = th.linalg.solve(M, L.transpose(1, 2) @ (msk * shading))
    n = th.nn.functional.normalize(x, dim=1)[..., 0].reshape(SIZE, SIZE, 3).cpu().numpy()
    np.save(f"{OUT}/rose__{name}.npy", n.astype(np.float32))
    Image.fromarray((n * 127.5 + 128).clip(0, 255).astype(np.uint8)).save(f"{OUT}/rose__{name}.png")
    subj = name.split("_")[0]
    for k, f in enumerate(frames):
        f.resize((768, 768), Image.BICUBIC).save(f"{OUT}/sets/{name}_rose__L_{k:02d}.png")
    Image.fromarray(masks[subj]).resize((768, 768), Image.NEAREST).save(f"{OUT}/sets/{name}_rose__mask.png")
    stats[name] = dict(seconds=round(time.time() - t0, 1))
    print(name, stats[name], flush=True)
del pipeline, unet
th.cuda.empty_cache()

# RoSE 음영 프레임을 SDM-UniPS로 (1024px)
os.chdir(f"{W}/SDM-UniPS")
for set_name in sorted({os.path.basename(p).split("__")[0] for p in glob.glob(f"{OUT}/sets/*__L_*.png")}):
    root = f"{W}/sdm_sets/{set_name}"
    d = f"{root}/{set_name}.data"
    os.makedirs(d, exist_ok=True)
    for k, p in enumerate(sorted(glob.glob(f"{OUT}/sets/{set_name}__L_*.png"))):
        im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        cv2.imwrite(f"{d}/L_{k:03d}.png", cv2.resize(im, (1536, 1536), interpolation=cv2.INTER_CUBIC))
    m = cv2.imread(f"{OUT}/sets/{set_name}__mask.png", cv2.IMREAD_UNCHANGED)
    cv2.imwrite(f"{d}/mask.png", cv2.resize(m, (1536, 1536), interpolation=cv2.INTER_NEAREST))
    if sh(f"PYTORCH_ALLOC_CONF=expandable_segments:True python sdm_unips/main.py --session_name sdm_{set_name} "
          f"--test_dir {root} --checkpoint {SDM_CKPT} --target normal --max_image_num 9 --scalable", check=False):
        shutil.copytree(f"sdm_{set_name}/results/{set_name}.data", f"{OUT}/sdm_results/{set_name}.data")
    shutil.rmtree(root, ignore_errors=True)
json.dump(stats, open(f"{OUT}/summary.json", "w"), indent=2)
print(json.dumps(stats, indent=2), flush=True)
