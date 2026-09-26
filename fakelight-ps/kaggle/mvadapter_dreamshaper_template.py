# MV-Adapter (I2MV, SDXL) + DreamShaper XL: 버니 앞면 1장 → 멀티뷰 6장.
# build_kernel.py가 __INPUT_PNG_B64__ 자리에 입력 이미지를 끼워 넣어서 Kaggle에 올린다.
import base64, io, json, os, subprocess, sys, time

def sh(cmd):
    print(f"\n$ {cmd}", flush=True)
    t = time.time()
    r = subprocess.run(cmd, shell=True)
    print(f"[exit={r.returncode} {time.time()-t:.0f}s]", flush=True)
    if r.returncode != 0:
        sys.exit(r.returncode)

W = "/kaggle/working"
OUT = f"{W}/out"
os.makedirs(OUT, exist_ok=True)

# mvadapter-env 노트북에서 설치가 검증된 버전 조합
sh("pip install -q 'diffusers==0.32.2' 'transformers==4.49.0' 'huggingface_hub==0.36.2' 'peft==0.14.0' "
   "accelerate einops jaxtyping typeguard trimesh omegaconf")
# I2MV는 래스터라이저를 안 쓰지만 mvadapter.utils가 import 시점에 nvdiffrast를 불러서 필요함
sh("pip install -q --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git")
sh(f"git clone -q https://github.com/huanngzh/MV-Adapter.git {W}/MV-Adapter "
   f"&& cd {W}/MV-Adapter && git checkout -q 4277e0018232bac82bb2c103caf0893cedb711be")
sys.path.insert(0, f"{W}/MV-Adapter")


def patch(path, old, new, count):
    src = open(path).read()
    assert src.count(old) == count, f"patch target not found {count}x in {path}: {old[:60]!r}"
    open(path, "w").write(src.replace(old, new))


# T4(14.5GB) OOM 패치: 원본은 레퍼런스 특징을 뷰 수 × CFG 2배로 복사해두고 .clone()까지 해서
# 768px 6뷰 기준 약 10GB를 먹는다. 어텐션은 이 값을 읽기만 하므로 [uncond, cond] 2벌만 들고 있다가
# 레이어마다 필요할 때 늘린다. 배치 순서가 [uncond×N, cond×N]이라 repeat_interleave 결과가 원본과 같다.
MVA = f"{W}/MV-Adapter/mvadapter"
patch(f"{MVA}/pipelines/pipeline_mvadapter_i2mv_sdxl.py",
      """            ref_hidden_states = {
                k: v.repeat_interleave(num_images_per_prompt, dim=0)
                for k, v in ref_hidden_states.items()
            }
""", "", 1)
patch(f"{MVA}/pipelines/pipeline_mvadapter_i2mv_sdxl.py",
      '"ref_hidden_states": {k: v.clone() for k, v in ref_hidden_states.items()},',
      '"ref_hidden_states": ref_hidden_states,', 1)
patch(f"{MVA}/models/attention_processor.py",
      "            reference_hidden_states = ref_hidden_states[self.name]\n",
      "            reference_hidden_states = ref_hidden_states[self.name]\n"
      "            reference_hidden_states = reference_hidden_states.repeat_interleave(\n"
      "                batch_size // reference_hidden_states.shape[0], dim=0)\n", 2)
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image
from mvadapter.utils import make_image_grid
from scripts.inference_i2mv_sdxl import run_pipeline

CFG = dict(
    base_model="Lykon/dreamshaper-xl-1-0",
    vae_model="madebyollin/sdxl-vae-fp16-fix",
    adapter_path="huanngzh/mv-adapter",
    azimuth_deg=[0, 45, 90, 180, 270, 315],
    text="a white clay sculpture of a bunny, studio lighting, high quality",
    num_inference_steps=50,
    guidance_scale=3.0,
    seed=42,
    height=768,
    width=768,
)
print(json.dumps(CFG, indent=2), flush=True)
print(torch.__version__, torch.cuda.get_device_name(0), flush=True)

INPUT_PNG_B64 = "__INPUT_PNG_B64__"
ref = Image.open(io.BytesIO(base64.b64decode(INPUT_PNG_B64)))  # RGBA, 투명 배경
ref.save(f"{OUT}/input_rgba.png")

def load_pipe(num_views):
    """scripts.inference_i2mv_sdxl.prepare_pipeline과 같은 순서인데,
    fp32(약 13GB) 대신 fp16 가중치를 받아서 다운로드/RAM을 절반으로 줄인다."""
    from diffusers import AutoencoderKL
    from mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl import MVAdapterI2MVSDXLPipeline
    from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler

    vae = AutoencoderKL.from_pretrained(CFG["vae_model"], torch_dtype=torch.float16)
    pipe = MVAdapterI2MVSDXLPipeline.from_pretrained(
        CFG["base_model"], vae=vae, variant="fp16", torch_dtype=torch.float16)
    pipe.scheduler = ShiftSNRScheduler.from_scheduler(
        pipe.scheduler, shift_mode="interpolated", shift_scale=8.0, scheduler_class=None)
    pipe.init_custom_adapter(num_views=num_views)
    pipe.load_custom_adapter(CFG["adapter_path"], weight_name="mvadapter_i2mv_sdxl.safetensors")
    pipe.to(device="cuda", dtype=torch.float16)
    pipe.cond_encoder.to(device="cuda", dtype=torch.float16)
    pipe.enable_vae_slicing()
    return pipe

t0 = time.time()
pipe = load_pipe(len(CFG["azimuth_deg"]))
t_load = time.time() - t0

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
images, reference = run_pipeline(
    pipe, num_views=len(CFG["azimuth_deg"]), text=CFG["text"], image=ref,
    height=CFG["height"], width=CFG["width"],
    num_inference_steps=CFG["num_inference_steps"], guidance_scale=CFG["guidance_scale"],
    seed=CFG["seed"], azimuth_deg=CFG["azimuth_deg"],
)
t_gen = time.time() - t0

reference.save(f"{OUT}/reference_preprocessed.png")
for az, im in zip(CFG["azimuth_deg"], images):
    im.save(f"{OUT}/view_az{az:03d}.png")
make_image_grid(images, rows=1).save(f"{OUT}/mv_grid.png")

stats = dict(CFG, load_seconds=round(t_load, 1), gen_seconds=round(t_gen, 1),
             peak_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
             gpu=torch.cuda.get_device_name(0))
json.dump(stats, open(f"{OUT}/stats.json", "w"), indent=2)
print(json.dumps(stats, indent=2), flush=True)
