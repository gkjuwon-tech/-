# SEVA 설정 점검: 공식 Gradio "Basic" 모드와 같은 설정으로 사진 1장 → 궤도 영상.
# 우리 첫 시도(img2img, 한 번에 생성, cfg 2.0, 90° 간격의 성긴 목표)와 공식 데모의 차이:
#   - 작업: img2trajvid_s-prob (사진 1장 + 미리 정한 궤도)
#   - 2단계 생성: 먼저 궤도 위 앵커 몇 장을 그리고(cfg 4.0), 앵커 사이를 촘촘히 채운다(cfg 2.0), guider 1,2
#   - 촘촘한 궤도: 공식 기본 80프레임 (T4/P100 시간 때문에 __NUM__프레임), 한 프레임씩 조금씩 돈다
#   - 궤도 cfg 4.0은 공식 데모의 orbit 기본값
# 입력은 흰 배경 정사각형으로 패딩 (SEVA는 기본으로 가운데를 잘라서, 세로로 긴 그림은 머리/발이 잘림).
import base64, glob, io, json, os, shutil, subprocess, sys, time

def sh(cmd, check=True):
    print(f"\n$ {cmd}", flush=True)
    t = time.time()
    r = subprocess.run(cmd, shell=True)
    print(f"[exit={r.returncode} {time.time()-t:.0f}s]", flush=True)
    if check and r.returncode != 0:
        sys.exit(r.returncode)
    return r.returncode == 0

W = "/kaggle/working"
TAG = "__TAG__"
NUM = int("__NUM__")
OUT = f"{W}/out/{TAG}"
os.makedirs(OUT, exist_ok=True)
os.environ["HF_TOKEN"] = "__HF_TOKEN__"
T0 = time.time()
stats = {}

sh(f"git clone -q --depth 1 https://github.com/Stability-AI/stable-virtual-camera.git {W}/seva")
sh("pip install -q roma tyro fire einops colorama splines kornia open-clip-torch diffusers 'imageio[ffmpeg]' "
   "huggingface-hub opencv-python-headless scipy 'gradio==5.17.0' viser ninja")
sh(f"cd {W}/seva && pip install -q --no-deps -e .")
os.chdir(f"{W}/seva")
for f, old, new in [("seva/modules/autoencoder.py", "stabilityai/stable-diffusion-2-1-base", "sd2-community/stable-diffusion-2-1-base"),
                    ("seva/utils.py", "model = Seva(SevaParams()).to(torch.bfloat16)", "model = Seva(SevaParams())"),
                    ("seva/modules/transformer.py", "with sdpa_kernel(SDPBackend.FLASH_ATTENTION):",
                     "with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):")]:
    s = open(f).read()
    assert s.count(old) == 1, (f, old)
    open(f, "w").write(s.replace(old, new))

# ---------- 입력 ----------
import numpy as np
from PIL import Image
src = "__SRC__"
if src.startswith("repo:"):                                   # SEVA 저장소의 공식 예제 그림
    im = Image.open(src[5:])
else:
    im = Image.open(io.BytesIO(base64.b64decode("__INPUT_PNG_B64__")))
im = im.convert("RGBA")
a = np.asarray(im).astype(np.float32) / 255
rgb = a[..., :3] * a[..., 3:] + (1 - a[..., 3:])
h, w = rgb.shape[:2]
n = max(h, w)
sq = np.ones((n, n, 3), np.float32)
sq[(n - h) // 2:(n - h) // 2 + h, (n - w) // 2:(n - w) // 2 + w] = rgb
os.makedirs(f"{W}/data", exist_ok=True)
Image.fromarray((sq * 255).round().astype(np.uint8)).resize((576, 576), Image.LANCZOS).save(f"{W}/data/{TAG}.png")
shutil.copy(f"{W}/data/{TAG}.png", f"{OUT}/input.png")

# ---------- 공식 Basic 설정으로 생성 ----------
for T in [21, 14]:
    t0 = time.time()
    ok = sh(f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python demo.py --data_path {W}/data --data_items {TAG}.png "
            f"--task img2trajvid_s-prob --replace_or_include_input True --traj_prior orbit --cfg 4.0,2.0 --guider 1,2 "
            f"--num_targets {NUM} --L_short 576 --use_traj_prior True --chunk_strategy interp --T {T}", check=False)
    outs = sorted(glob.glob(f"work_dirs/demo/img2trajvid_s-prob/{TAG}/samples-rgb/*.png"))
    stats[f"T{T}"] = dict(ok=ok, n=len(outs), seconds=round(time.time() - t0, 1))
    print(stats[f"T{T}"], flush=True)
    if ok and outs:
        break
    shutil.rmtree("work_dirs", ignore_errors=True)
res = f"work_dirs/demo/img2trajvid_s-prob/{TAG}"
if os.path.isdir(res):
    shutil.copytree(res, f"{OUT}/result", dirs_exist_ok=True)
stats["total_seconds"] = round(time.time() - T0, 1)
json.dump(stats, open(f"{OUT}/stats.json", "w"), indent=2)
sh(f"find {OUT} -maxdepth 2 | head -30", check=False)
print(json.dumps(stats, indent=2), flush=True)
