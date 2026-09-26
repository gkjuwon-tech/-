# 멀티뷰 가짜 조명: LightSwitch (ICCV 2025, 재질 인식 멀티뷰 재조명) → 뷰마다 SDM-UniPS.
# 입력 데이터셋: images/*.png + views.json (뷰마다 OpenCV c2w, 가로 FOV). 카메라를 알고 있으니 COLMAP 텍스트를 직접 쓴다.
# 조명: 구 전체에 고르게 퍼진 방향광 K개를 각각 HDR 환경맵(작은 밝은 원반 + 약한 환경광)으로 만들어
#       모든 뷰를 같은 세계 조명으로 한 번에 재조명한다 → 뷰 사이 조명이 일관된 광도 스테레오 세트.
# 공식 코드에서 고친 곳: SD2.1 base 저장소 이름(sd2-community 미러), 환경맵 .npy 로드, 한 번 로드한 모델로 환경맵 K개 반복.
import glob, json, os, shutil, subprocess, sys, time

def sh(cmd, check=True):
    print(f"\n$ {cmd}", flush=True)
    t = time.time()
    r = subprocess.run(cmd, shell=True)
    print(f"[exit={r.returncode} {time.time()-t:.0f}s]", flush=True)
    if check and r.returncode != 0:
        sys.exit(r.returncode)
    return r.returncode == 0

TAG = "__TAG__"
NUM_LIGHTS = int("__NUM_LIGHTS__")
MODEL = "__MODEL__"            # thebluser/lightswitch 또는 thebluser/lightswitch-multi-fov
RES = int("__RES__")           # 재조명 해상도 (정사각)
W = "/kaggle/working"
OUT = f"{W}/out/{TAG}"
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
stats = {}

sh("pip install -q -U 'diffusers>=0.31' accelerate imageio plyfile rembg onnxruntime")
sh(f"git clone -q --depth 1 https://github.com/yehonathanlitman/LightSwitch.git {W}/LightSwitch")
sh(f"git clone -q https://github.com/satoshi-ikehata/SDM-UniPS-CVPR2023.git {W}/SDM-UniPS")
sh(f"curl -sSL -o {W}/sdm_ckpt.zip 'https://www.dropbox.com/s/yu8h6g0zp07mumd/checkpoint.zip?dl=1' "
   f"&& cd {W} && unzip -q -o sdm_ckpt.zip -d sdm_ckpt")
SDM_CKPT = os.path.dirname(os.path.dirname(glob.glob(f"{W}/sdm_ckpt/**/normal/*.pytmodel", recursive=True)[0]))

import numpy as np
from PIL import Image

# ---------- 1) 장면 폴더: 이미지, 마스크, COLMAP 텍스트 ----------
SRC = os.path.dirname(glob.glob("/kaggle/input/**/views.json", recursive=True)[0])
spec = json.load(open(f"{SRC}/views.json"))
scene = f"{W}/scene"
for d in ["images", "masks", "sparse/0"]:
    os.makedirs(f"{scene}/{d}", exist_ok=True)
from rembg import remove, new_session
sess = new_session("isnet-general-use")
cams, imgs = [], []
for i, v in enumerate(spec["views"]):
    im = Image.open(f"{SRC}/images/{v['name']}").convert("RGB").resize((RES, RES), Image.LANCZOS)
    m = np.asarray(remove(im, session=sess, only_mask=True)) > 127
    name = os.path.splitext(v["name"])[0] + ".png"
    im.save(f"{scene}/images/{name}")
    Image.fromarray((m * 255).astype(np.uint8)).save(f"{scene}/masks/{name}")
    c2w = np.array(v["c2w_opencv"])
    w2c = np.linalg.inv(c2w)
    R, t = w2c[:3, :3], w2c[:3, 3]
    # 회전행렬 → 쿼터니언 (w, x, y, z)
    qw = np.sqrt(max(0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    qx = np.copysign(np.sqrt(max(0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    qy = np.copysign(np.sqrt(max(0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    qz = np.copysign(np.sqrt(max(0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    fx = RES / 2 / np.tan(np.radians(v["fovx_deg"]) / 2)
    cams.append(f"{i + 1} PINHOLE {RES} {RES} {fx} {fx} {RES / 2} {RES / 2}")
    imgs.append(f"{i + 1} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} {i + 1} {name}\n")
open(f"{scene}/sparse/0/cameras.txt", "w").write("\n".join(cams) + "\n")
open(f"{scene}/sparse/0/images.txt", "w").write("\n".join(imgs) + "\n")
shutil.copytree(f"{scene}/images", f"{OUT}/input_views")
shutil.copytree(f"{scene}/masks", f"{OUT}/masks")

# ---------- 2) 방향광 환경맵 (LightSwitch의 방향 임베딩과 같은 좌표로 그린다) ----------
os.chdir(f"{W}/LightSwitch")
sys.path.insert(0, f"{W}/LightSwitch")
from dataset_colmap import generate_directional_embeddings
dirs = generate_directional_embeddings((256, 512))                     # (256, 512, 3), 픽셀별 세계 방향
k = np.arange(NUM_LIGHTS) + 0.5                                       # 피보나치 구: 구 전체에 고르게
z = 1 - 2 * k / NUM_LIGHTS
phi = np.pi * (1 + 5 ** 0.5) * k
L = np.stack([np.sqrt(1 - z ** 2) * np.cos(phi), np.sqrt(1 - z ** 2) * np.sin(phi), z], 1)
os.makedirs(f"{W}/envmaps", exist_ok=True)
env_paths = []
for j, l in enumerate(L):
    cosang = dirs @ l
    env = 0.02 + 60.0 * np.exp((cosang - 1) / (1 - np.cos(np.radians(4))) )  # 반경 약 4°의 밝은 원반
    env = np.repeat(env[..., None], 3, 2).astype(np.float32)
    p = f"{W}/envmaps/L_{j:02d}.npy"
    np.save(p, env)
    env_paths.append(p)
np.save(f"{OUT}/light_dirs_world.npy", L)

# ---------- 3) 공식 코드 패치 ----------
s = open("produce_gs_relightings.py").read()
s = s.replace("stabilityai/stable-diffusion-2-1-base", "sd2-community/stable-diffusion-2-1-base")
# model_index.json이 CLIPFeatureExtractor 클래스를 이름으로 부르므로 transformers에 별칭도 달아둔다
s = ("import transformers\nif not hasattr(transformers, 'CLIPFeatureExtractor'):\n"
     "    transformers.CLIPFeatureExtractor = transformers.CLIPImageProcessor\n") + s
old = "    produce_colmap(accelerate, args, args.scene_dir, pipeline, stable_material, weight_dtype=weight_dtype, generator=generator)"
assert s.count(old) == 1
s = s.replace(old, "    for _env in args.envmap_path.split(','):\n        args.envmap_path = _env\n"
                   "        produce_colmap(accelerate, args, args.scene_dir, pipeline, stable_material, weight_dtype=weight_dtype, generator=generator)")
s = s.replace('choices=["thebluser/lightswitch", "thebluser/lightswitch-multi-fov"],)', ')')
open("produce_gs_relightings.py", "w").write(s)
d = open("dataset_colmap.py").read()
old = "        self.envmap = imageio.imread(envmap_path)[..., :3]"
assert d.count(old) == 1
d = d.replace(old, "        self.envmap = (np.load(envmap_path) if envmap_path.endswith('.npy') else imageio.imread(envmap_path))[..., :3]")
open("dataset_colmap.py", "w").write(d)

# 신버전 transformers에서 CLIPFeatureExtractor가 사라졌다 → CLIPImageProcessor로 바꿔 부른다
for f in glob.glob(f"{W}/LightSwitch/**/*.py", recursive=True):
    t = open(f).read()
    if "CLIPFeatureExtractor" in t:
        open(f, "w").write(t.replace("CLIPFeatureExtractor", "CLIPImageProcessor"))

# ---------- 4) 재조명 (T4 두 장에 나눠서) ----------
t0 = time.time()
ok = sh(f"accelerate launch --num_processes 2 --multi_gpu --mixed_precision fp16 produce_gs_relightings.py "
        f"--pretrained_model {MODEL} --scene_dir {scene} --image_dir_name images --downsample 1 "
        f"--envmap_path {','.join(env_paths)}", check=False)
if not ok:
    ok = sh(f"python produce_gs_relightings.py --pretrained_model {MODEL} --scene_dir {scene} --image_dir_name images "
            f"--downsample 1 --envmap_path {','.join(env_paths)}", check=False)
stats["relight"] = dict(ok=ok, seconds=round(time.time() - t0, 1))
res_root = glob.glob(f"{W}/LightSwitch/relighting_outputs/rm_*/scene")
print("relight outputs:", res_root, flush=True)
if res_root:
    res_root = res_root[0]
    for sub in ["albedo", "orm"]:
        if os.path.isdir(f"{res_root}/{sub}"):
            shutil.copytree(f"{res_root}/{sub}", f"{OUT}/{sub}")
    for j in range(NUM_LIGHTS):
        src_dir = f"{res_root}/L_{j:02d}/images"
        if os.path.isdir(src_dir):
            shutil.copytree(src_dir, f"{OUT}/relit/L_{j:02d}")

# ---------- 5) 뷰마다 SDM-UniPS (조명 K장) ----------
os.chdir(f"{W}/SDM-UniPS")
import cv2
for v in spec["views"]:
    name = os.path.splitext(v["name"])[0] + ".png"
    view = os.path.splitext(name)[0]
    root = f"{W}/sdm_sets/{view}"
    dd = f"{root}/{view}.data"
    os.makedirs(dd, exist_ok=True)
    n = 0
    for j in range(NUM_LIGHTS):
        p = f"{OUT}/relit/L_{j:02d}/{name}"
        if os.path.exists(p):
            im = cv2.imread(p)
            cv2.imwrite(f"{dd}/L_{n:03d}.png", cv2.resize(im, (1024, 1024), interpolation=cv2.INTER_CUBIC))
            n += 1
    if n < 3:
        stats[f"sdm_{view}"] = f"only {n} images"
        continue
    m = cv2.imread(f"{scene}/masks/{name}", cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(f"{dd}/mask.png", cv2.resize(m, (1024, 1024), interpolation=cv2.INTER_NEAREST))
    for kk in [n, 16, 12, 9]:
        if kk > n:
            continue
        if sh(f"PYTORCH_ALLOC_CONF=expandable_segments:True python sdm_unips/main.py --session_name sdm_{view} "
              f"--test_dir {root} --checkpoint {SDM_CKPT} --target normal --max_image_num {kk} --scalable", check=False):
            os.makedirs(f"{OUT}/normals", exist_ok=True)
            shutil.copy(f"sdm_{view}/results/{view}.data/normal.png", f"{OUT}/normals/{name}")
            stats[f"sdm_{view}"] = kk
            break
    shutil.rmtree(root, ignore_errors=True)

stats["total_seconds"] = round(time.time() - T0, 1)
json.dump(stats, open(f"{OUT}/stats.json", "w"), indent=2)
sh(f"find {OUT} -maxdepth 2 | head -60", check=False)
print(json.dumps(stats, indent=2), flush=True)
