# SDM-UniPS만 다시 돌린다: 이미 만든 IC-Light 리라이팅(데이터셋 flyjw12/fakelight-relights-v2)을 세트로 묶어서 푼다.
# 파일 이름 규칙: {세트}__L_XX.png, {세트}__mask.png (예: jugg_s1__L_03.png)
# 세트마다 별도 프로세스로 돌리고, 메모리가 터지면 이미지 장수를 줄여 재시도한다 (SDM-UniPS는 장수에 비례해 메모리를 먹는다).
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
sh(f"git clone -q https://github.com/satoshi-ikehata/SDM-UniPS-CVPR2023.git {W}/SDM-UniPS")
sh(f"curl -sSL -o {W}/sdm_ckpt.zip 'https://www.dropbox.com/s/yu8h6g0zp07mumd/checkpoint.zip?dl=1' "
   f"&& cd {W} && unzip -q -o sdm_ckpt.zip -d sdm_ckpt")
SDM_CKPT = os.path.dirname(os.path.dirname(glob.glob(f"{W}/sdm_ckpt/**/normal/*.pytmodel", recursive=True)[0]))

from PIL import Image

relit_dir = next(d for d in glob.glob("/kaggle/input/**/", recursive=True) if glob.glob(f"{d}/jugg_s*__L_00.png"))
real_dir = next(d for d in glob.glob("/kaggle/input/**/", recursive=True) if os.path.exists(f"{d}/real__L_00.png"))
seeds = sorted({re.match(r"(\w+?)_s(\d+)__", os.path.basename(p)).group(2)
                for p in glob.glob(f"{relit_dir}/jugg_s*__L_00.png")})
print("seeds:", seeds, flush=True)


def build_set(dst, groups, scale=1):
    """groups: [(폴더, 접두사)], 이미지를 L_000.. 로 이어 붙인다."""
    os.makedirs(dst, exist_ok=True)
    k = 0
    for d, pre in groups:
        for p in sorted(glob.glob(f"{d}/{pre}L_*.png")):
            im = Image.open(p)
            if scale != 1:
                im = im.resize((im.width * scale, im.height * scale), Image.BICUBIC)
            im.save(f"{dst}/L_{k:03d}.png"); k += 1
    m = Image.open(f"{groups[0][0]}/{groups[0][1]}mask.png")
    if scale != 1:
        m = m.resize((m.width * scale, m.height * scale), Image.NEAREST)
    m.save(f"{dst}/mask.png")
    return k


SETS = {}
for subject in ("jugg", "render"):
    for s in seeds:
        SETS[f"{subject}_s{s}"] = ([(relit_dir, f"{subject}_s{s}__")], 1)
    SETS[f"{subject}_all"] = ([(relit_dir, f"{subject}_s{s}__") for s in seeds], 1)
    # 1024px 내부 해상도 변형은 메모리 때문에 첫 시드 16장만
    SETS[f"{subject}_s{seeds[0]}_x2"] = ([(relit_dir, f"{subject}_s{seeds[0]}__")], 2)
SETS["real_lights"] = ([(real_dir, "real__")], 1)

os.chdir(f"{W}/SDM-UniPS")
env = "PYTORCH_ALLOC_CONF=expandable_segments:True"
summary = {}
for name, (groups, scale) in SETS.items():
    root = f"{W}/sdm_sets/{name}"
    n = build_set(f"{root}/{name}.data", groups, scale)
    ok = False
    for k in [n, 32, 24, 16]:
        if k > n:
            continue
        t0 = time.time()
        ok = sh(f"{env} python sdm_unips/main.py --session_name sdm_{name} --test_dir {root} "
                f"--checkpoint {SDM_CKPT} --target normal_and_brdf --max_image_num {k} --scalable", check=False)
        if ok:
            summary[name] = dict(images=k, of=n, scale=scale, seconds=round(time.time() - t0, 1))
            shutil.copytree(f"sdm_{name}/results/{name}.data", f"{OUT}/sdm_results/{name}.data")
            break
        print(f"{name}: failed with {k} images, retrying smaller", flush=True)
    if not ok:
        summary[name] = dict(error="all attempts failed", of=n)

json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2), flush=True)
