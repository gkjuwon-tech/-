# 풀이기 교체 실험: SDM-UniPS(CVPR 2023) 대신 LINO-UniPS(ICLR 2026)로 같은 리라이팅 세트를 푼다.
# 입력은 데이터셋 flyjw12/fakelight-relights-v2 ({세트}__L_XX.png, {세트}__mask.png)와
# flyjw12/fakelight-inputs (real__*, 진짜 조명 8장). 출력은 out/lino_results/{세트}.data/normal.png
# (SDM-UniPS와 같은 (n+1)/2 RGB 형식)라서 scripts/score_sdm.py로 그대로 채점한다.
import gc, glob, json, os, re, subprocess, sys, time, traceback

def sh(cmd, check=True):
    print(f"\n$ {cmd}", flush=True)
    t = time.time()
    r = subprocess.run(cmd, shell=True)
    print(f"[exit={r.returncode} {time.time()-t:.0f}s]", flush=True)
    if check and r.returncode != 0:
        sys.exit(r.returncode)
    return r.returncode == 0

W = "/kaggle/working"
OUT = f"{W}/out/lino_results"
os.makedirs(OUT, exist_ok=True)
LINO = f"{W}/LINO_UniPS"
sh(f"git clone -q --depth 1 https://github.com/houyuanchen111/LINO_UniPS.git {LINO}")
sh("pip install -q pytorch_lightning lightning hydra-core kornia PyWavelets torchmetrics rich omegaconf einops")
# pyexr/utils3d는 HDRI·학습 코드에서만 쓰인다. 설치가 안 되면 빈 모듈로 대신한다
for mod, pkg in [("pyexr", "pyexr"), ("utils3d", "git+https://github.com/EasternJournalist/utils3d.git")]:
    if not sh(f"pip install -q {pkg} && python -c 'import {mod}'", check=False):
        os.makedirs(f"{W}/stubs", exist_ok=True)
        open(f"{W}/stubs/{mod}.py", "w").write("# stub: not needed for normal prediction\n")
        print(f"using stub for {mod}", flush=True)
sys.path[:0] = [f"{W}/stubs", LINO]

import numpy as np
import torch
from PIL import Image

os.chdir(LINO)
model = torch.hub.load(LINO, "lino_unips", source="local", pretrained=True, task_name="Real")
# 모델 내부가 입력을 bf16으로 강제 변환하므로(model_step) 가중치도 bf16으로 맞춘다. T4에서 bf16 matmul 동작은 확인됨
model = model.to("cuda", dtype=torch.bfloat16).eval()
from src.data import DemoData


def predict(imgs, mask):
    """imgs: RGB uint8 배열 리스트, mask: HxW uint8. hubconf.Predictor.predict와 같되 roi는 정수로 유지
    (bf16로 바꾸면 256 넘는 좌표가 반올림되어 크롭이 어긋난다)."""
    data = DemoData([(im, None) for im in imgs], np.stack([mask] * 3, -1))[0]
    batch = {}
    for k, v in data.items():
        if k == "roi":
            batch[k] = torch.tensor(np.asarray(v), dtype=torch.long, device="cuda")[None]
        else:
            batch[k] = torch.tensor(np.asarray(v), dtype=torch.bfloat16, device="cuda")[None]
    with torch.no_grad():
        return model(batch)  # (H, W, 3) 원본 해상도, 마스크 밖 0


relit_dir = next(d for d in glob.glob("/kaggle/input/**/", recursive=True) if glob.glob(f"{d}/jugg_s*__L_00.png"))
real_dir = next(d for d in glob.glob("/kaggle/input/**/", recursive=True) if os.path.exists(f"{d}/real__L_00.png"))
seeds = sorted({re.match(r"\w+?_s(\d+)__", os.path.basename(p)).group(1)
                for p in glob.glob(f"{relit_dir}/jugg_s*__L_00.png")})


def load(groups, scale=1):
    imgs = []
    for d, pre in groups:
        for p in sorted(glob.glob(f"{d}/{pre}L_*.png")):
            im = Image.open(p).convert("RGB")
            if scale != 1:
                im = im.resize((im.width * scale, im.height * scale), Image.BICUBIC)
            imgs.append(np.asarray(im))
    m = Image.open(f"{groups[0][0]}/{groups[0][1]}mask.png").convert("L")
    if scale != 1:
        m = m.resize((m.width * scale, m.height * scale), Image.NEAREST)
    return imgs, np.asarray(m)


SETS = {}
for subject in ("jugg", "render"):
    for s in seeds:
        SETS[f"{subject}_s{s}"] = ([(relit_dir, f"{subject}_s{s}__")], 1)
    SETS[f"{subject}_all"] = ([(relit_dir, f"{subject}_s{s}__") for s in seeds], 1)
    SETS[f"{subject}_s{seeds[0]}_x2"] = ([(relit_dir, f"{subject}_s{seeds[0]}__")], 2)
SETS["real_lights"] = ([(real_dir, "real__")], 1)

summary = {}
rng = np.random.default_rng(0)
for name, (groups, scale) in SETS.items():
    imgs, mask = load(groups, scale)
    done = False
    for k in [len(imgs), 32, 24, 16]:
        if k > len(imgs):
            continue
        sub = imgs if k == len(imgs) else [imgs[i] for i in sorted(rng.choice(len(imgs), k, replace=False))]
        try:
            t0 = time.time()
            torch.cuda.reset_peak_memory_stats()
            n = predict(sub, mask)
            d = f"{OUT}/{name}.data"
            os.makedirs(d, exist_ok=True)
            if scale != 1:  # 채점은 원래 해상도에서
                n = np.stack([np.asarray(Image.fromarray(n[..., c].astype(np.float32), "F").resize(
                    (n.shape[1] // scale, n.shape[0] // scale), Image.BILINEAR)) for c in range(3)], 2)
                n /= np.linalg.norm(n, axis=2, keepdims=True) + 1e-12
            Image.fromarray(((n * 0.5 + 0.5) * 255 * (np.abs(n).sum(2, keepdims=True) > 0)).clip(0, 255)
                            .astype(np.uint8)).save(f"{d}/normal.png")
            summary[name] = dict(images=k, of=len(imgs), scale=scale, seconds=round(time.time() - t0, 1),
                                 peak_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
            print(name, summary[name], flush=True)
            done = True
            break
        except torch.OutOfMemoryError:
            print(f"{name}: OOM with {k} images, retrying smaller", flush=True)
        except Exception as e:
            traceback.print_exc()
            summary[name] = dict(error=repr(e))
            break
        gc.collect(); torch.cuda.empty_cache()
    if not done and name not in summary:
        summary[name] = dict(error="OOM at all sizes")
    gc.collect(); torch.cuda.empty_cache()

json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2), flush=True)
