# 비교 기준선: 이미지 1장에서 노멀을 직접 예측하는 모델들 (Marigold v1.1, StableNormal, DSINE)
# 우리 파이프라인과 같은 입력(쥬거너트 정면 뷰, 원본 렌더)으로 돌려서 로컬에서 같은 정답으로 채점한다.
# 모델마다 노멀 좌표계 부호가 달라서 원본 예측(.npy)을 그대로 저장하고, 부호 규약은 채점 단계에서
# 대조군(render)으로 한 번 정해 쥬거너트 세트에 똑같이 적용한다.
import glob, json, os, subprocess, sys, time, traceback

def sh(cmd, check=True):
    print(f"\n$ {cmd}", flush=True)
    t = time.time()
    r = subprocess.run(cmd, shell=True)
    print(f"[exit={r.returncode} {time.time()-t:.0f}s]", flush=True)
    if check and r.returncode != 0:
        sys.exit(r.returncode)

W = "/kaggle/working"
OUT = f"{W}/out/normal_baselines"
os.makedirs(OUT, exist_ok=True)
# StableNormal은 diffusers.models.controlnet(0.31에서 이동)을 import해서 구버전이 필요하다. DSINE은 geffnet 필요.
# Marigold(최신 diffusers)는 1차 실행에서 이미 끝났으므로 MODELS로 이번에 돌릴 것만 고른다.
MODELS = __BASE_MODELS__
if "marigold_v1_1" in MODELS:
    sh("pip install -q -U diffusers transformers accelerate")
    sh("pip uninstall -q -y torchao", check=False)
else:
    sh("pip install -q 'diffusers==0.30.3' geffnet")

import numpy as np
import torch
from PIL import Image

src_dir = next(d for d in glob.glob("/kaggle/input/**/", recursive=True)
               if os.path.exists(os.path.join(d, "jugg_front.png")))
inputs = {"jugg": Image.open(f"{src_dir}/jugg_front.png").convert("RGB")}
rgba = np.asarray(Image.open(f"{src_dir}/render_front_rgba.png")).astype(np.float32) / 255.0
inputs["render"] = Image.fromarray(((rgba[..., :3] * rgba[..., 3:] + 0.5 * (1 - rgba[..., 3:])) * 255).astype(np.uint8))
stats = {}


def save(model, name, n):
    n = np.asarray(n, dtype=np.float32)
    np.save(f"{OUT}/{model}__{name}.npy", n)
    Image.fromarray(((n * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)).save(f"{OUT}/{model}__{name}.png")


def run(model, factory):
    if model not in MODELS:
        return
    try:
        t0 = time.time()
        fn = factory()
        for name, img in inputs.items():
            save(model, name, fn(img))
        stats[model] = dict(seconds=round(time.time() - t0, 1))
        print(f"{model} ok", flush=True)
    except Exception as e:
        traceback.print_exc()
        stats[model] = dict(error=repr(e))
    torch.cuda.empty_cache()


# ---------- Marigold normals v1.1 ----------
def marigold():
    from diffusers import MarigoldNormalsPipeline
    pipe = MarigoldNormalsPipeline.from_pretrained(
        "prs-eth/marigold-normals-v1-1", variant="fp16", torch_dtype=torch.float16).to("cuda")
    def f(img):
        out = pipe(img, ensemble_size=10, generator=torch.Generator("cuda").manual_seed(0))
        return out.prediction[0]  # (H, W, 3), [-1, 1]
    return f

run("marigold_v1_1", marigold)

# ---------- StableNormal ----------
def stablenormal():
    m = torch.hub.load("Stable-X/StableNormal", "StableNormal", trust_repo=True)
    def f(img):
        vis = m(img)  # PIL, (n+1)/2 시각화
        return np.asarray(vis.resize(img.size), dtype=np.float32) / 255.0 * 2 - 1
    return f

run("stablenormal", stablenormal)

# ---------- DSINE ----------
def dsine():
    m = torch.hub.load("hugoycj/DSINE-hub", "DSINE", trust_repo=True)
    def f(img):
        n = m.infer_pil(img)[0]  # (3, H, W)
        return n.permute(1, 2, 0).float().cpu().numpy()
    return f

run("dsine", dsine)

json.dump(stats, open(f"{OUT}/stats.json", "w"), indent=2)
print(json.dumps(stats, indent=2), flush=True)
