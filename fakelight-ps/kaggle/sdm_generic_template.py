# 범용 SDM-UniPS 실행기: 마운트된 데이터셋에서 {세트}__L_XX.png / {세트}__mask.png 파일을 세트별로 묶어 푼다.
# 8/16비트 모두 그대로 쓰고, 2배 업스케일(SDM-UniPS 내부 1024px)로 푼다.
# COMBOS: 여러 세트를 이어 붙인 합본 세트 {이름: [접두사 정규식, ...]}
import glob, json, os, re, shutil, subprocess, sys, time

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
UPSCALE = 2
COMBOS = json.loads(__BASE_MODELS__[0]) if __BASE_MODELS__ else {}  # build_kernel --models 에 JSON 한 줄로 전달
sh(f"git clone -q https://github.com/satoshi-ikehata/SDM-UniPS-CVPR2023.git {W}/SDM-UniPS")
sh(f"curl -sSL -o {W}/sdm_ckpt.zip 'https://www.dropbox.com/s/yu8h6g0zp07mumd/checkpoint.zip?dl=1' "
   f"&& cd {W} && unzip -q -o sdm_ckpt.zip -d sdm_ckpt")
SDM_CKPT = os.path.dirname(os.path.dirname(glob.glob(f"{W}/sdm_ckpt/**/normal/*.pytmodel", recursive=True)[0]))

import cv2

files = glob.glob("/kaggle/input/**/*__L_*.png", recursive=True)
by_set = {}
for p in files:
    by_set.setdefault(os.path.basename(p).split("__")[0], []).append(p)
masks = {s: next(iter(glob.glob(f"/kaggle/input/**/{s}__mask.png", recursive=True)), None) for s in by_set}
SETS = {s: (sorted(ps), masks[s]) for s, ps in by_set.items()}
for name, pats in COMBOS.items():
    members = [s for s in sorted(by_set) if any(re.fullmatch(p, s) for p in pats)]
    SETS[name] = ([p for s in members for p in sorted(by_set[s])], masks[members[0]])
print({k: len(v[0]) for k, v in SETS.items()}, flush=True)


def build(dst, paths, mask_path):
    os.makedirs(dst, exist_ok=True)
    for i, p in enumerate(paths):
        im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        im = cv2.resize(im, (im.shape[1] * UPSCALE, im.shape[0] * UPSCALE), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(f"{dst}/L_{i:03d}.png", im)
    m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    cv2.imwrite(f"{dst}/mask.png", cv2.resize(m, (m.shape[1] * UPSCALE, m.shape[0] * UPSCALE),
                                              interpolation=cv2.INTER_NEAREST))


os.chdir(f"{W}/SDM-UniPS")
env = "PYTORCH_ALLOC_CONF=expandable_segments:True"
summary = {}
for name, (paths, mask_path) in SETS.items():
    root = f"{W}/sdm_sets/{name}"
    build(f"{root}/{name}.data", paths, mask_path)
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
