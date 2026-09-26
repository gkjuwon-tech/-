# 멀티뷰 영상: Wan2.2-Fun-5B-Control-Camera (알리바바 PAI, Apache 2.0) — 입력 사진 1장 + 프레임별 카메라 궤도.
# Veo와 달리 첫 프레임이 입력 사진 그대로이고, 카메라를 우리가 프레임마다 정한다 (zoo14 14뷰를 지나는 궤도).
# 공식 predict_v2v_control_camera_5b.py의 설정 줄만 바꿔서 실행한다. Lightning에서 확인한 것:
#   - VideoX-Fun 최신판의 오타 (xfuser 없을 때 xFUserLongContextAttention = None) → 이름 바로잡기
# T4 대응: bf16 미지원이라 fp16, 메모리 모드는 model_cpu_offload 먼저, 실패하면 sequential_cpu_offload.
# Kaggle RAM 29GB: T5-XXL은 별도 프로세스에서 프롬프트만 인코딩하고 내려놓는다.
import base64, glob, json, os, re, shutil, subprocess, sys, time

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
TRAJS = "__TRAJS__".split(",")                 # 예: ring 또는 up,dn
SIZE = "__SIZE__"                              # 예: 512x512
OUT = f"{W}/out/{TAG}"
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
stats = {}

sh("nvidia-smi -L; df -h /tmp /kaggle/working | tail -2; free -g | head -2", check=False)
sh("git clone -q --depth 1 https://github.com/aigc-apps/VideoX-Fun.git /tmp/VideoX-Fun")
os.chdir("/tmp/VideoX-Fun")
# torch는 Kaggle 것을 그대로 쓰고 나머지만 설치 (torch를 다시 깔면 CUDA가 어긋날 수 있음)
req = [l.strip() for l in open("requirements.txt") if l.strip() and not l.startswith("#") and not l.startswith("torch")]
open("req_notorch.txt", "w").write("\n".join(req))
sh("pip install -q -r req_notorch.txt", check=False)
s = open("videox_fun/dist/fuser.py").read()
open("videox_fun/dist/fuser.py", "w").write(s.replace("    xFUserLongContextAttention = None", "    xFuserLongContextAttention = None"))
MODEL = "/tmp/Wan2.2-Fun-5B-Control-Camera"
sh(f"HF_HUB_ENABLE_HF_TRANSFER=1 python -c \"from huggingface_hub import snapshot_download; "
   f"snapshot_download('alibaba-pai/Wan2.2-Fun-5B-Control-Camera', local_dir='{MODEL}')\" || "
   f"python -c \"from huggingface_hub import snapshot_download; snapshot_download('alibaba-pai/Wan2.2-Fun-5B-Control-Camera', local_dir='{MODEL}')\"")
os.makedirs("models/Diffusion_Transformer", exist_ok=True)
os.symlink(MODEL, "models/Diffusion_Transformer/Wan2.2-Fun-5B-Control-Camera")

# ---------- 입력 파일 풀기 (입력 사진, 궤도 파일들, 프롬프트) ----------
os.makedirs("fl", exist_ok=True)
for name, b64 in json.loads('__INPUT_FILES__').items():
    open(os.path.join("fl", name.split("/", 1)[1]), "wb").write(base64.b64decode(b64))
prompt = open("fl/prompt.txt").read().strip()
sh("ls -la fl")

# ---------- 텍스트 인코딩은 따로 (T5-XXL 11GB + 영상 모델 10GB를 RAM에 동시에 올리면 Kaggle 29GB가 터짐) ----------
NEG = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
       "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")
open("fl/neg.txt", "w").write(NEG)
open("encode_prompts.py", "w").write(r"""
import os, torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from videox_fun.models import WanT5EncoderModel
cfg = OmegaConf.load("config/wan2.2/wan_civitai_5b.yaml")
M = "models/Diffusion_Transformer/Wan2.2-Fun-5B-Control-Camera"
tok = AutoTokenizer.from_pretrained(os.path.join(M, cfg['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')))
te = WanT5EncoderModel.from_pretrained(os.path.join(M, cfg['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
                                       additional_kwargs=OmegaConf.to_container(cfg['text_encoder_kwargs']),
                                       low_cpu_mem_usage=True, torch_dtype=torch.bfloat16).eval()
out = {}
for key, path in [("pos", "fl/prompt.txt"), ("neg", "fl/neg.txt")]:
    t = tok([open(path).read().strip()], padding="max_length", max_length=512, truncation=True, add_special_tokens=True, return_tensors="pt")
    n = int(t.attention_mask.sum())
    with torch.no_grad():
        e = te(t.input_ids, attention_mask=t.attention_mask)[0][0, :n]
    out[key] = e.float()
    print(key, tuple(e.shape), float(e.float().abs().mean()), flush=True)
torch.save(out, "fl/prompt_embeds.pt")
""")
sh("python encode_prompts.py")

src = open("examples/wan2.2_fun/predict_v2v_control_camera_5b.py").read()
# 본 생성에서는 T5를 안 올리고 저장해둔 임베딩을 쓴다 (파이프라인은 text_encoder.dtype만 참조)
i0 = src.index("text_encoder = WanT5EncoderModel.from_pretrained(")
i1 = src.index("text_encoder = text_encoder.eval()") + len("text_encoder = text_encoder.eval()")
src = src[:i0] + """class _NoTextEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("_d", torch.zeros(1, dtype=weight_dtype))
    @property
    def dtype(self):
        return weight_dtype
text_encoder = _NoTextEncoder()
_E = torch.load("fl/prompt_embeds.pt")""" + src[i1:]
assert src.count("    sample = pipeline(\n        prompt, \n") == 1
src = src.replace("    sample = pipeline(\n        prompt, \n", "    sample = pipeline(\n        prompt, prompt_embeds=[_E['pos'].to(device, weight_dtype)], negative_prompt_embeds=[_E['neg'].to(device, weight_dtype)],\n")
h, w = SIZE.split("x")
for traj in TRAJS:
    for mem in ["model_cpu_offload", "sequential_cpu_offload"]:
        t0 = time.time()
        code = src
        for pat, rep in {
            r'^GPU_memory_mode\s*=.*$': f'GPU_memory_mode     = "{mem}"',
            r'^sample_size\s*=.*$': f'sample_size         = [{h}, {w}]',
            r'^weight_dtype\s*=.*$': 'weight_dtype            = torch.float16',
            r'^control_camera_txt\s*=.*$': f'control_camera_txt      = "fl/zoo_{traj}.txt"',
            r'^start_image\s*=.*$': 'start_image             = "fl/input.png"',
            r'^prompt\s*=.*$': 'prompt                  = ' + repr(prompt),
            r'^save_path\s*=.*$': f'save_path               = "samples/{traj}"',
        }.items():
            code, n = re.subn(pat, rep, code, count=1, flags=re.M)
            assert n == 1, pat
        open(f"run_{traj}.py", "w").write(code)
        ok = sh(f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python run_{traj}.py", check=False)
        vids = sorted(glob.glob(f"samples/{traj}/*.mp4"))
        stats[f"{traj}_{mem}"] = dict(ok=ok, n=len(vids), seconds=round(time.time() - t0, 1))
        print(traj, mem, stats[f"{traj}_{mem}"], flush=True)
        if ok and vids:
            shutil.copy(vids[-1], f"{OUT}/{TAG}_{traj}.mp4")
            break
        shutil.rmtree(f"samples/{traj}", ignore_errors=True)
    json.dump(stats, open(f"{OUT}/stats.json", "w"), indent=2)

for f in glob.glob("fl/zoo_*.json"):
    shutil.copy(f, OUT)
stats["total_seconds"] = round(time.time() - T0, 1)
json.dump(stats, open(f"{OUT}/stats.json", "w"), indent=2)
print(json.dumps(stats, indent=2), flush=True)
