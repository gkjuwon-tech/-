# 멀티뷰 오디션: WAVE (ICCV 2025, 학습 없이 워핑으로 시점 일관성을 잡는 기법) + MegaScenes NVS 체크포인트.
# 입력: 버니 정면 렌더 1장 (RGBA). 공식 inference.py를 그대로 쓰되 세 군데만 고친다:
#   1) 체크포인트: 저자 로컬 경로로 accelerator.load_state 하던 부분 → 공개 model.safetensors를 직접 로드
#   2) 깊이: 물체 마스크 밖(배경) 깊이를 0으로 둬서 워핑 메쉬에 배경이 섞이지 않게 함
#   3) 카메라 경로: 공식 궤도(±30° 안쪽, 장면용) + 물체 중심을 도는 궤도(0, ±45, ±90, ±135, 180°) 두 번 실행
# 해상도는 모델 고정값 256px.
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
OUT = f"{W}/out/wave"
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
stats = {}

sh("apt-get install -y -qq libegl1 libgl1 libgles2 libosmesa6 > /dev/null 2>&1", check=False)
sh("pip install -q 'pytorch-lightning==1.9.5' 'omegaconf==2.3.0' einops kornia taming-transformers-rom1504 "
   "git+https://github.com/openai/CLIP.git trimesh pyrender 'pyglet<2' icecream ipdb imageio pytz accelerate "
   "scikit-image safetensors")
sh(f"git clone -q --depth 1 https://github.com/jwoo-park0/WAVE.git {W}/WAVE")
sh(f"git clone -q --depth 1 https://github.com/LiheYoung/Depth-Anything.git {W}/Depth-Anything")
# Depth-Anything은 DINOv2를 작업 폴더 기준 torchhub/에서 불러온다 → WAVE 폴더에 연결
sh(f"ln -s {W}/Depth-Anything/torchhub {W}/WAVE/torchhub && ls {W}/WAVE/torchhub")
ck = f"{W}/WAVE/configs/warp_plus_pose/iter_112000"
os.makedirs(ck, exist_ok=True)
sh(f"curl -sSL -o {ck}/model.safetensors "
   "https://megascenes.s3.amazonaws.com/nvs_checkpoints/warp_plus_pose/iter_112000/model.safetensors")
sh(f"ls -la {ck}")

# ---------- 입력: RGBA → 흰 배경 RGB + 마스크 ----------
from PIL import Image
import numpy as np
rgba = Image.open(io.BytesIO(base64.b64decode("__INPUT_PNG_B64__"))).convert("RGBA")
a = np.asarray(rgba).astype(np.float32) / 255
rgb = a[..., :3] * a[..., 3:] + (1 - a[..., 3:])
Image.fromarray((rgb * 255).round().astype(np.uint8)).save(f"{W}/input.png")
Image.fromarray(((a[..., 3] > 0.5) * 255).astype(np.uint8)).save(f"{W}/input_mask.png")
shutil.copy(f"{W}/input.png", f"{OUT}/input.png")

# ---------- inference.py 패치 ----------
os.chdir(f"{W}/WAVE")
src = open("inference.py").read()
reps = [
    # 1) 체크포인트 직접 로드
    ("        accelerator.load_state('/nfs/home/wldn1677/nvs/megascene/configs/warp_plus_pose/iter_112000')#join(args.exp_dir, resume_folder))",
     "        from safetensors.torch import load_file\n"
     "        sd = load_file(join(args.exp_dir, 'warp_plus_pose', resume_folder, 'model.safetensors'))\n"
     "        tgt = model.module if hasattr(model, 'module') else model\n"
     "        res = tgt.load_state_dict(sd, strict=False)\n"
     "        print('ckpt loaded: missing', len(res.missing_keys), res.missing_keys[:5], 'unexpected', len(res.unexpected_keys), res.unexpected_keys[:5], flush=True)"),
    # 2) 배경 깊이 제거 + 물체 궤도
    ("    latent_depthmap = resize(depthmap, (32,32))",
     "    _m = np.asarray(Image.open(os.environ['WAVE_MASK']).convert('L').resize((w, h), Image.NEAREST)) > 127\n"
     "    depthmap = depthmap * _m\n"
     "    global OBJ_DEPTH\n"
     "    OBJ_DEPTH = float(np.median(depthmap[_m]))\n"
     "    print('object median depth', OBJ_DEPTH, flush=True)\n"
     "    latent_depthmap = resize(depthmap, (32,32), order=0)"),
    ("    orbitposes = get_orbit_poses()",
     "    orbitposes = get_orbit_poses() if os.environ.get('WAVE_MODE') == 'default' else object_orbit(OBJ_DEPTH, YAWS)"),
]
for old, new in reps:
    assert src.count(old) == 1, old
    src = src.replace(old, new)
helper = '''
YAWS = [0, 45, 90, 135, 180, -135, -90, -45]
OBJ_DEPTH = 1.0


def object_orbit(dc, yaws):
    """물체 중심 (0, 0, dc)를 고정점으로 y축 회전하는 카메라들 (공식 코드와 같은 w2c → c2w 규약)."""
    p = np.array([0.0, 0.0, dc])
    poses = []
    for yaw in yaws:
        th = np.radians(yaw)
        R = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
        E = np.eye(4)
        E[:3, :3] = R
        E[:3, 3] = p - R @ p
        poses.append(np.linalg.inv(E))
    return poses

'''
src = src.replace("def setup_model():", helper + "def setup_model():", 1)
open("inference.py", "w").write(src)
# pyrender는 import할 때 뷰어(pyglet 창)까지 불러서 화면 없는 서버에서 죽는다 → 뷰어 import만 제거
_pr = subprocess.run("python -c 'import importlib.util as u; print(u.find_spec(\"pyrender\").submodule_search_locations[0])'",
                     shell=True, capture_output=True, text=True).stdout.strip()
_init = open(f"{_pr}/__init__.py").read().replace("from .viewer import Viewer", "Viewer = None")
open(f"{_pr}/__init__.py", "w").write(_init)
# pytorch_lightning 신버전에서 사라진 import 정리
for f in glob.glob("ldm/**/*.py", recursive=True) + glob.glob("dataloader/*.py"):
    s = open(f).read()
    s2 = s.replace("from pytorch_lightning.utilities.distributed import rank_zero_only",
                   "from pytorch_lightning.utilities import rank_zero_only")
    if s2 != s:
        open(f, "w").write(s2)

env = dict(os.environ, PYTHONPATH=f"{W}/Depth-Anything:{W}/WAVE", WAVE_MASK=f"{W}/input_mask.png",
           PYOPENGL_PLATFORM="egl", PYGLET_HEADLESS="True")
for mode in ["object", "default"]:
    t0 = time.time()
    env["WAVE_MODE"] = mode
    ok = subprocess.run(f"python inference.py -e configs/ -r 112000 --config_name config -i {W}/input.png "
                        f"-s {OUT}/{mode}", shell=True, env=env).returncode == 0
    if not ok and mode == "object":   # EGL이 안 되면 OSMesa로 한 번 더
        env["PYOPENGL_PLATFORM"] = "osmesa"
        ok = subprocess.run(f"python inference.py -e configs/ -r 112000 --config_name config -i {W}/input.png "
                            f"-s {OUT}/{mode}", shell=True, env=env).returncode == 0
    stats[mode] = dict(ok=ok, seconds=round(time.time() - t0, 1))
    print(mode, stats[mode], flush=True)

stats["yaws_object"] = [0, 45, 90, 135, 180, -135, -90, -45]
stats["total_seconds"] = round(time.time() - T0, 1)
json.dump(stats, open(f"{OUT}/stats.json", "w"), indent=2)
sh(f"find {OUT} -maxdepth 3 | head -80", check=False)
print(json.dumps(stats, indent=2), flush=True)
