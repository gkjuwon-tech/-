# 멀티뷰 오디션 후보: Qwen-Image-Edit-2511 (2025-12) + fal Multiple-Angles LoRA + Lightning 4-step LoRA.
# 정면 버니 1장 → 7개 방위각 뷰 (0°는 입력). T4에 맞추려고:
#   - 트랜스포머(20B)는 GGUF 4비트로 cuda:0 (메모리 부족하면 3비트로 자동 후퇴)
#   - 텍스트 인코더(Qwen2.5-VL-7B)는 bitsandbytes 8비트로 cuda:1 (GPU가 1장이면 CPU)
# 입력은 Kaggle 데이터셋(flyjw12/fakelight-inputs)의 render_front_rgba.png.
import gc, glob, json, os, subprocess, sys, time

def sh(cmd, check=True):
    print(f"\n$ {cmd}", flush=True)
    t = time.time()
    r = subprocess.run(cmd, shell=True)
    print(f"[exit={r.returncode} {time.time()-t:.0f}s]", flush=True)
    if check and r.returncode != 0:
        sys.exit(r.returncode)

W = "/kaggle/working"
OUT = f"{W}/out/Qwen-Image-Edit-2511-MultiAngles"
os.makedirs(OUT, exist_ok=True)
sh("pip install -q -U diffusers transformers accelerate peft gguf bitsandbytes")
# Kaggle 이미지의 torchao 0.10이 최신 peft와 충돌해서 LoRA 주입이 실패한다 (peft는 0.16+ 요구). 안 쓰니까 제거
sh("pip uninstall -q -y torchao", check=False)

import numpy as np
import torch
from PIL import Image

ngpu = torch.cuda.device_count()
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(ngpu)], torch.__version__, flush=True)
MAIN = torch.device("cuda:0")

# T4(sm75)는 bf16 텐서코어가 없다. 돌아가기만 하면 bf16을 쓴다 (Qwen-Image는 fp16에서 오버플로 이슈가 알려져 있음)
try:
    a = torch.randn(64, 64, device=MAIN, dtype=torch.bfloat16)
    assert torch.isfinite(a @ a).all()
    DTYPE = torch.bfloat16
except Exception as e:
    print("bf16 matmul failed, falling back to fp16:", e, flush=True)
    DTYPE = torch.float16
print("compute dtype:", DTYPE, flush=True)

CFG = dict(
    base="Qwen/Qwen-Image-Edit-2511",
    gguf_repo="unsloth/Qwen-Image-Edit-2511-GGUF",
    gguf_candidates=["qwen-image-edit-2511-Q4_0.gguf", "qwen-image-edit-2511-Q3_K_M.gguf"],
    lightning=("lightx2v/Qwen-Image-Edit-2511-Lightning", "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"),
    angles=("fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA", "qwen-image-edit-2511-multiple-angles-lora.safetensors"),
    angles_scale=0.9,
    steps=4,
    true_cfg_scale=1.0,
    size=1024,
    seed=42,
    # LoRA README의 8방위 표기. 0°(front view)는 입력 그대로라 생성하지 않는다
    views={45: "front-right quarter view", 90: "right side view", 135: "back-right quarter view",
           180: "back view", 225: "back-left quarter view", 270: "left side view",
           315: "front-left quarter view"},
    suffix="eye-level shot medium shot",
)
print(json.dumps(CFG, indent=2), flush=True)

# ---------- 입력: 흰 배경 정면 버니 ----------
src = next(p for p in glob.glob("/kaggle/input/**/render_front_rgba.png", recursive=True))
rgba = np.asarray(Image.open(src)).astype(np.float32) / 255.0
rgb = rgba[..., :3] * rgba[..., 3:] + (1 - rgba[..., 3:])
front = Image.fromarray((rgb * 255).astype(np.uint8)).resize((CFG["size"], CFG["size"]), Image.LANCZOS)
front.save(f"{OUT}/view_az000.png")

# ---------- 텍스트 인코더 ----------
from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
t0 = time.time()
if ngpu >= 2:
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        CFG["base"], subfolder="text_encoder", torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map={"": 1})
else:
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        CFG["base"], subfolder="text_encoder", torch_dtype=torch.bfloat16)
TE_DEV = next(text_encoder.parameters()).device
print(f"text encoder on {TE_DEV} ({time.time()-t0:.0f}s)", flush=True)

from diffusers import GGUFQuantizationConfig, QwenImageEditPlusPipeline, QwenImageTransformer2DModel


def build_pipe(gguf_file):
    transformer = QwenImageTransformer2DModel.from_single_file(
        f"https://huggingface.co/{CFG['gguf_repo']}/blob/main/{gguf_file}",
        quantization_config=GGUFQuantizationConfig(compute_dtype=DTYPE), torch_dtype=DTYPE,
        config=CFG["base"], subfolder="transformer")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        CFG["base"], transformer=transformer, text_encoder=text_encoder, torch_dtype=DTYPE)
    pipe.transformer.to(MAIN)
    pipe.vae.to(MAIN)
    pipe.load_lora_weights(CFG["lightning"][0], weight_name=CFG["lightning"][1], adapter_name="lightning")
    pipe.load_lora_weights(CFG["angles"][0], weight_name=CFG["angles"][1], adapter_name="angles")
    pipe.set_adapters(["lightning", "angles"], adapter_weights=[1.0, CFG["angles_scale"]])

    # 텍스트 인코더는 다른 장치에 있으니, 임베딩만 그쪽에서 뽑아 cuda:0으로 옮긴다
    orig = pipe._get_qwen_prompt_embeds
    def embeds_on_main(prompt=None, image=None, device=None, dtype=None):
        e, m = orig(prompt, image, TE_DEV, None)
        return e.to(MAIN, DTYPE), m.to(MAIN)
    pipe._get_qwen_prompt_embeds = embeds_on_main
    # 컴포넌트 순서상 text_encoder가 먼저라 실행 장치를 cuda:0으로 고정
    type(pipe)._execution_device = property(lambda self: MAIN)
    return pipe


def generate(pipe, az):
    prompt = f"<sks> {CFG['views'][az]} {CFG['suffix']}"
    g = torch.Generator(device=MAIN).manual_seed(CFG["seed"])
    return pipe(image=[front], prompt=prompt, negative_prompt=" ", true_cfg_scale=CFG["true_cfg_scale"],
                num_inference_steps=CFG["steps"], height=CFG["size"], width=CFG["size"],
                generator=g).images[0]


stats = dict(CFG, views={str(k): v for k, v in CFG["views"].items()}, gpus=ngpu, dtype=str(DTYPE))
pipe = None
for gguf_file in CFG["gguf_candidates"]:
    try:
        t0 = time.time()
        pipe = build_pipe(gguf_file)
        stats.update(gguf=gguf_file, load_seconds=round(time.time() - t0, 1))
        torch.cuda.reset_peak_memory_stats(MAIN)
        t0 = time.time()
        first_az = next(iter(CFG["views"]))
        generate(pipe, first_az).save(f"{OUT}/view_az{first_az:03d}.png")
        print(f"{gguf_file}: first view ok ({time.time()-t0:.0f}s)", flush=True)
        break
    except torch.OutOfMemoryError as e:
        print(f"OOM with {gguf_file}, trying smaller quant: {e}", flush=True)
        pipe = None; gc.collect(); torch.cuda.empty_cache()
assert pipe is not None, "all GGUF candidates OOM"

times = []
for az in list(CFG["views"])[1:]:
    t0 = time.time()
    generate(pipe, az).save(f"{OUT}/view_az{az:03d}.png")
    times.append(time.time() - t0)
    print(f"az {az} done ({times[-1]:.0f}s)", flush=True)

# MV-Adapter 결과와 같은 6개 각도로 그리드 (MV-Adapter 방위각 부호는 채점 단계에서 정답과 대조해 확인)
order = [0, 45, 90, 180, 270, 315]
tiles = [Image.open(f"{OUT}/view_az{a:03d}.png").convert("RGB").resize((768, 768)) for a in order]
grid = Image.new("RGB", (768 * len(tiles), 768))
for i, t in enumerate(tiles):
    grid.paste(t, (768 * i, 0))
grid.save(f"{OUT}/mv_grid.png")
stats.update(gen_seconds_per_view=round(float(np.mean(times)), 1) if times else None,
             gen_seconds=round(float(np.sum(times)), 1),
             peak_mem_gb=round(torch.cuda.max_memory_allocated(MAIN) / 1e9, 2),
             base_model="Qwen-Image-Edit-2511 + MultiAngles LoRA")
json.dump(stats, open(f"{OUT}/stats.json", "w"), indent=2)
print(json.dumps(stats, indent=2), flush=True)
