# 광도 스테레오 교체 시험: SDM-UniPS → GeoUniPS (AAAI 2026, VGGT 기하 사전지식 + 다중 조명 단서).
# 재조명 이미지는 v1 원정에서 만든 것을 그대로 쓰고 (데이터셋 fakelight-relight-sets-v1), 풀이기만 바꾼다.
# 세트마다 SDM-UniPS와 같은 레이아웃 ({root}/{set}.data/L_*.png + mask.png)으로 넣고 노멀을 저장한다.
# 물체 여러 개(어려운 것 + 쉬운 것)를 한 번에 돌려서 개선이 한 물체에만 맞춘 게 아닌지 같이 본다.
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
OUT = f"{W}/out/geounips"
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
stats = {}

sh("pip install -q einops opencv-python-headless huggingface_hub")
sh(f"git clone -q https://github.com/marcotam2002/geounips.git {W}/geounips")
sh(f"cd {W}/geounips && ls -R | head -50 && (cat requirements.txt 2>/dev/null || true)")
from huggingface_hub import snapshot_download
snapshot_download("marcotam2002/geounips", local_dir=f"{W}/geounips/checkpoint")
sh(f"ls -la {W}/geounips/checkpoint")
sh(f"cd {W}/geounips && python geo_unips/main.py --help", check=False)

import cv2
import numpy as np

SRC = glob.glob("/kaggle/input/**/harvest/ic_rot__L_00.png", recursive=True)[0].rsplit("/harvest/", 1)[0]
OBJS = ["harvest", "cat", "ball", "bear", "reading"]
SETS = ["ic_rot", "ic_plain", "nlr_all"]
os.chdir(f"{W}/geounips")
for obj in OBJS:
    for s in SETS:
        paths = sorted(glob.glob(f"{SRC}/{obj}/{s}__L_*.png"))
        root = f"{W}/geo_sets/{obj}_{s}"
        d = f"{root}/{s}.data"
        os.makedirs(d, exist_ok=True)
        for i, p in enumerate(paths):
            im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            cv2.imwrite(f"{d}/L_{i:03d}.png", cv2.resize(im, (1024, 1024), interpolation=cv2.INTER_CUBIC))
        m = cv2.imread(f"{SRC}/{obj}/{s}__mask.png", cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(f"{d}/mask.png", cv2.resize(m, (1024, 1024), interpolation=cv2.INTER_NEAREST))
        key = f"{obj}_{s}"
        t0 = time.time()
        for res, k in [(1024, len(paths)), (1024, 16), (512, len(paths)), (512, 16), (512, 9)]:
            if k > len(paths):
                continue
            sess = f"geo_{key}"
            ok = sh(f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python geo_unips/main.py --session_name {sess} "
                    f"--test_dir {root} --checkpoint checkpoint --max_image_num {k} --max_image_res {res} --scalable",
                    check=False)
            outs = glob.glob(f"{sess}/**/normal.png", recursive=True)
            if ok and outs:
                os.makedirs(f"{OUT}/{obj}", exist_ok=True)
                shutil.copy(outs[0], f"{OUT}/{obj}/{s}_normal.png")
                stats[key] = dict(res=res, k=k, seconds=round(time.time() - t0, 1))
                break
            shutil.rmtree(sess, ignore_errors=True)
        else:
            stats[key] = "failed"
        print(key, stats[key], flush=True)
        shutil.rmtree(root, ignore_errors=True)

stats["total_seconds"] = round(time.time() - T0, 1)
json.dump(stats, open(f"{OUT}/stats.json", "w"), indent=2)
print(json.dumps(stats, indent=2), flush=True)
