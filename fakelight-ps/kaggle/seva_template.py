# 멀티뷰 생성: Stable Virtual Camera (Stability AI, 2025, 1.3B, 576px) — 사진 1장 → 물체를 도는 20개 시점.
# 장면(입력 사진 + 카메라)은 로컬에서 scripts/seva_scene.py로 만들어 끼워 넣는다 (__INPUT_FILES__).
# 로컬 CPU에서 1스텝으로 전체 경로를 먼저 검증했고, 그때 찾은 것들을 여기서 고친다:
#   - numpy==1.24.4 고정 → Kaggle(파이썬 3.12)에 휠이 없음: 의존성은 고정 없이 직접 설치하고 seva는 --no-deps
#   - VAE를 막힌 stabilityai/stable-diffusion-2-1-base에서 부름 → sd2-community 미러
#   - 모델을 bfloat16으로 올림 → T4는 bf16 미지원: fp32로 두고 autocast(fp16)로 계산
#   - 어텐션을 FlashAttention으로 고정 → T4에 없음 (CPU 검증으로는 못 잡음): 다른 커널도 허용
# 21장(입력 1 + 목표 20)을 한 번에 생성해 시점 일관성을 최대로. 메모리가 모자라면 T=11(두 번 나눠 생성)로 재시도.
import base64, glob, json, os, shutil, subprocess, sys, time

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
OUT = f"{W}/out/{TAG}"
os.makedirs(OUT, exist_ok=True)
os.environ["HF_TOKEN"] = "__HF_TOKEN__"
T0 = time.time()
stats = {}

# ---------- 장면 파일 풀기 ----------
scene = f"{W}/data/{TAG}"
for name, b64 in json.loads('__INPUT_FILES__').items():
    base = name.split("/", 1)[1]
    p = os.path.join(scene, "images", base) if base.endswith(".png") else os.path.join(scene, base)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(base64.b64decode(b64))
sh(f"find {scene}")
shutil.copy(f"{scene}/views.json", f"{OUT}/views.json")

# ---------- 설치 ----------
sh(f"git clone -q --depth 1 https://github.com/Stability-AI/stable-virtual-camera.git {W}/seva")
sh("pip install -q roma tyro fire einops colorama splines kornia open-clip-torch diffusers 'imageio[ffmpeg]' "
   "huggingface-hub opencv-python-headless scipy 'gradio==5.17.0' viser ninja")
sh(f"cd {W}/seva && pip install -q --no-deps -e .")
os.chdir(f"{W}/seva")
for f, old, new in [("seva/modules/autoencoder.py", "stabilityai/stable-diffusion-2-1-base", "sd2-community/stable-diffusion-2-1-base"),
                    ("seva/utils.py", "model = Seva(SevaParams()).to(torch.bfloat16)", "model = Seva(SevaParams())"),
                    # FlashAttention만 허용 → T4(sm75)에는 없음: 메모리 효율 커널과 기본 커널도 허용
                    ("seva/modules/transformer.py", "with sdpa_kernel(SDPBackend.FLASH_ATTENTION):",
                     "with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):")]:
    s = open(f).read()
    assert s.count(old) == 1, (f, old)
    open(f, "w").write(s.replace(old, new))

# ---------- 생성 ----------
# 사진 1장은 물체까지 거리를 모른다 (SEVA 문서도 인정하는 스케일 모호성). SEVA는 첫 카메라의 거리를
# camera_scale로 맞추는데, 모델이 그림에서 느끼는 거리와 다르면 엉뚱한 점을 축으로 돌아서 조각난 그림이 나온다.
# 그래서 camera_scale을 몇 개 훑어서 전부 저장한다 (정답을 보고 고르지 않고, 결과를 나란히 비교).
views = json.load(open(f"{scene}/views.json"))["views"]
for cs in [float(x) for x in "__SCALES__".split(",")]:
    tag = f"cs{cs:g}"
    for T in [int(x) for x in "__TS__".split(",")]:
        t0 = time.time()
        ok = sh(f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python demo.py --data_path {W}/data --data_items {TAG} "
                f"--task img2img --num_inputs 1 --T {T} --camera_scale {cs} --video_save_fps 10 --save_subdir {tag}",
                check=False)
        outs = sorted(glob.glob(f"work_dirs/demo/img2img/{tag}/{TAG}/samples-rgb/*.png"))
        stats[f"{tag}_T{T}"] = dict(ok=ok, n=len(outs), seconds=round(time.time() - t0, 1))
        print(tag, T, stats[f"{tag}_T{T}"], flush=True)
        if ok and len(outs) >= len(views) - 1:
            break
        shutil.rmtree(f"work_dirs/demo/img2img/{tag}", ignore_errors=True)
    # samples-rgb/000.. = 목표 1.. (split 순서). 입력은 0번.
    vd = f"{OUT}/{tag}/views"
    os.makedirs(vd, exist_ok=True)
    shutil.copy(f"{scene}/images/input.png", f"{vd}/{views[0]['name']}")
    for k, p in enumerate(sorted(glob.glob(f"work_dirs/demo/img2img/{tag}/{TAG}/samples-rgb/*.png"))):
        if k + 1 < len(views):
            shutil.copy(p, f"{vd}/{views[k + 1]['name']}")

stats["total_seconds"] = round(time.time() - T0, 1)
json.dump(stats, open(f"{OUT}/stats.json", "w"), indent=2)
print(json.dumps(stats, indent=2), flush=True)
