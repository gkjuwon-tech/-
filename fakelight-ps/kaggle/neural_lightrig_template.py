# IC-Light 대체 후보: Neural LightRig (CVPR 2025) 멀티라이트 확산 모델.
# 물리 기반 렌더링으로 학습했고, 조명 9개의 위치가 고정/공개되어 있고(θ=i·45°, φ=[30,60,…,0]°),
# 9장을 한 번에 픽셀 정렬로 생성한다 (3×3 격자 768px → 한 장 256px).
# 1) 멀티라이트 이미지 생성 (시드 3개) → 2) SDM-UniPS로 풀기 (2배 업스케일) → 3) Neural LightRig 자체 노멀도 저장(기준선)
# HF 동의 모델이라 토큰이 필요하다. 토큰은 빌드 시에만 끼워 넣고(build/는 git 제외) 저장소에는 남기지 않는다.
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
os.makedirs(OUT, exist_ok=True)
os.environ["HF_TOKEN"] = "__HF_TOKEN__"
NLR = f"{W}/Neural-LightRig"

sh("pip install -q 'diffusers==0.31.0' 'transformers==4.46.3' 'huggingface_hub==0.36.2' accelerate "
   "omegaconf einops pytorch-lightning megfile wandb")
sh(f"git clone -q --depth 1 https://github.com/ZexinHe/Neural-LightRig.git {NLR}")
# SD 2.1 공식 저장소가 HF에서 내려가서 sd2-community 미러로 바꾼다 (본체 가중치는 mld.pt에 들어 있음)
for f in glob.glob(f"{NLR}/mld/**/*.py", recursive=True) + glob.glob(f"{NLR}/mld/configs/*.yaml"):
    s = open(f).read()
    s2 = s.replace("stabilityai/stable-diffusion-2-1-unclip", "sd2-community/stable-diffusion-2-1-unclip") \
          .replace('"stabilityai/stable-diffusion-2-1"', '"sd2-community/stable-diffusion-2-1"')
    if s2 != s:
        open(f, "w").write(s2)
        print("patched", f, flush=True)
sh(f"git clone -q https://github.com/satoshi-ikehata/SDM-UniPS-CVPR2023.git {W}/SDM-UniPS")
sh(f"curl -sSL -o {W}/sdm_ckpt.zip 'https://www.dropbox.com/s/yu8h6g0zp07mumd/checkpoint.zip?dl=1' "
   f"&& cd {W} && unzip -q -o sdm_ckpt.zip -d sdm_ckpt")
SDM_CKPT = os.path.dirname(os.path.dirname(glob.glob(f"{W}/sdm_ckpt/**/normal/*.pytmodel", recursive=True)[0]))

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, NLR)
os.chdir(NLR)
from huggingface_hub import snapshot_download
from inference import prepare_stage1_mld, prepare_stage2_recon
from utils.vis import replace_bg_preserving_alpha

ckpt = snapshot_download(repo_id="zxhezexin/neural-lightrig-mld-and-recon", local_dir=f"{W}/nlr_ckpt")
model1 = prepare_stage1_mld(cfg_path="./mld/configs/infer.yaml", ckpt_path=os.path.join(ckpt, "mld.pt"))
model2 = prepare_stage2_recon(ckpt_path=os.path.join(ckpt, "recon"))

CFG = dict(seeds=[511, 1, 2], input_res=512, cfg_scale=2.0, cfg_rescale=0.7, steps=75, frame=768,
           thetas_deg=[i * 45 for i in range(9)], phis_deg=[30, 60, 30, 60, 30, 60, 30, 60, 0])
json.dump(CFG, open(f"{OUT}/nlr_config.json", "w"), indent=2)

inputs_dir = next(d for d in glob.glob("/kaggle/input/**/", recursive=True) if os.path.exists(f"{d}/jugg_front.png"))
relit_dir = next(d for d in glob.glob("/kaggle/input/**/", recursive=True) if os.path.exists(f"{d}/jugg_s1__mask.png"))
jugg = np.asarray(Image.open(f"{inputs_dir}/jugg_front.png").convert("RGB"))
jugg_mask = np.asarray(Image.open(f"{relit_dir}/jugg_s1__mask.png").convert("L"))  # IC-Light 배경 제거 마스크 (정답 아님)
render = np.asarray(Image.open(f"{inputs_dir}/render_front_rgba.png").convert("RGBA"))
subjects = {
    "jugg": Image.fromarray(np.dstack([jugg, jugg_mask])),
    "render": Image.fromarray(render),
}

stats = {}
for name, rgba in subjects.items():
    F = CFG["frame"]
    rgba_in = rgba.resize((CFG["input_res"], CFG["input_res"]))
    mask_full = (np.asarray(rgba.convert("RGBA").resize((F, F)))[..., 3] > 127).astype(np.uint8) * 255
    for seed in CFG["seeds"]:
        torch.manual_seed(seed); np.random.seed(seed)
        t0 = time.time()
        grid = model1.pipeline(image=replace_bg_preserving_alpha(rgba_in, 255).convert("RGB"),
                               guidance_scale=CFG["cfg_scale"], guidance_rescale=CFG["cfg_rescale"],
                               num_inference_steps=CFG["steps"]).images[0]
        grid.save(f"{OUT}/{name}_nlr_s{seed}_grid.png")
        g = np.asarray(grid.convert("RGB"))
        T = g.shape[0] // 3
        prefix = f"{name}_nlr_s{seed}"
        os.makedirs(f"{OUT}/sets", exist_ok=True)
        for i in range(9):  # 행 우선: i = row*3 + col
            tile = g[(i // 3) * T:(i // 3 + 1) * T, (i % 3) * T:(i % 3 + 1) * T]
            Image.fromarray(tile).resize((F, F), Image.BICUBIC).save(f"{OUT}/sets/{prefix}__L_{i:02d}.png")
        Image.fromarray(mask_full).save(f"{OUT}/sets/{prefix}__mask.png")
        stats[prefix] = dict(seconds=round(time.time() - t0, 1))
        if seed == CFG["seeds"][0]:
            _, _, img_normal = model2.predict(input_image=replace_bg_preserving_alpha(rgba_in, 0), ref_image=grid)
            img_normal.resize((F, F), Image.BILINEAR).save(f"{OUT}/nlr_normal_{name}.png")
        print(prefix, stats[prefix], flush=True)
del model1, model2
torch.cuda.empty_cache()

# ---------- SDM-UniPS ----------
by_set = {}
for p in sorted(glob.glob(f"{OUT}/sets/*__L_*.png")):
    by_set.setdefault(os.path.basename(p).split("__")[0], []).append(p)
SETS = {s: ps for s, ps in by_set.items()}
for name in subjects:
    SETS[f"{name}_nlr_all"] = [p for s in sorted(by_set) if s.startswith(f"{name}_nlr_s") for p in by_set[s]]
os.chdir(f"{W}/SDM-UniPS")
env = "PYTORCH_ALLOC_CONF=expandable_segments:True"
for set_name, paths in SETS.items():
    root = f"{W}/sdm_sets/{set_name}"
    d = f"{root}/{set_name}.data"
    os.makedirs(d, exist_ok=True)
    for i, p in enumerate(paths):
        im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        cv2.imwrite(f"{d}/L_{i:03d}.png", cv2.resize(im, (im.shape[1] * 2, im.shape[0] * 2), interpolation=cv2.INTER_CUBIC))
    subj = set_name.split("_")[0]
    m = cv2.imread(glob.glob(f"{OUT}/sets/{subj}_nlr_s*__mask.png")[0], cv2.IMREAD_UNCHANGED)
    cv2.imwrite(f"{d}/mask.png", cv2.resize(m, (m.shape[1] * 2, m.shape[0] * 2), interpolation=cv2.INTER_NEAREST))
    ok = False
    for k in [len(paths), 18, 9]:
        if k > len(paths):
            continue
        if sh(f"{env} python sdm_unips/main.py --session_name sdm_{set_name} --test_dir {root} "
              f"--checkpoint {SDM_CKPT} --target normal --max_image_num {k} --scalable", check=False):
            shutil.copytree(f"sdm_{set_name}/results/{set_name}.data", f"{OUT}/sdm_results/{set_name}.data")
            stats[f"sdm_{set_name}"] = dict(images=k, of=len(paths))
            ok = True
            break
    if not ok:
        stats[f"sdm_{set_name}"] = dict(error="failed")
    shutil.rmtree(root, ignore_errors=True)

json.dump(stats, open(f"{OUT}/summary.json", "w"), indent=2)
print(json.dumps(stats, indent=2), flush=True)
