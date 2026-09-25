"""Kaggle T4 job: SV3D_p orbits from hero images. Pushed by push_sv3d.sh.
Outputs /kaggle/working/out/<hero>/<orbit>/<azimuth>.png plus poses.json."""
import atexit, glob, json, os, shutil, subprocess, sys, time
if os.environ.get("SV3D_WORKER") is None: atexit.register(lambda: (shutil.rmtree("/kaggle/working/gm", ignore_errors=True), shutil.rmtree("/kaggle/working/tmp", ignore_errors=True)))
t_start = time.time()
sh = lambda c: subprocess.run(c, shell=True, check=True)
if os.environ.get("SV3D_WORKER") is None: sh("pip install -q omegaconf einops fire kornia open-clip-torch pytorch-lightning invisible-watermark rembg onnxruntime-gpu 'imageio[ffmpeg]' timm ftfy regex git+https://github.com/openai/CLIP.git")
if os.environ.get("SV3D_WORKER") is None and not os.path.exists("/kaggle/working/gm"):
    sh("git clone -q --depth 1 https://github.com/Stability-AI/generative-models /kaggle/working/gm")
def locate(name):
    for root, _, files in os.walk("/kaggle/input"):
        if name in files:
            return os.path.join(root, name)
    raise SystemExit("missing " + name)
token = open(locate("hf_token.txt")).read().strip()
from huggingface_hub import hf_hub_download
os.makedirs("/kaggle/working/gm/checkpoints", exist_ok=True)
# keep the 9 GB checkpoint out of /kaggle/working (that folder is the job's output)
ck = hf_hub_download("stabilityai/sv3d", "sv3d_p.safetensors", local_dir="/tmp/ckpt", token=token)
if not os.path.exists("/kaggle/working/gm/checkpoints/sv3d_p.safetensors"):
    os.symlink(ck, "/kaggle/working/gm/checkpoints/sv3d_p.safetensors")
os.chdir("/kaggle/working/gm"); sys.path.insert(0, "/kaggle/working/gm")
if os.environ.get("SV3D_WORKER") is None:
    import torch as _t
    ngpu = max(_t.cuda.device_count(), 1)
    procs = [subprocess.Popen([sys.executable, os.path.abspath(__file__)], env=dict(os.environ, SV3D_WORKER=str(i), SV3D_NWORK=str(ngpu), CUDA_VISIBLE_DEVICES=str(i))) for i in range(ngpu)]
    rc = [p.wait() for p in procs]
    print("workers exit", rc, "total minutes", round((time.time() - t_start) / 60, 1))
    sys.exit(0)
import numpy as np, torch, imageio
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])
import scripts.sampling.simple_video_sample as svs
captured = {}
_orig = imageio.mimwrite
def grab(path, frames, *a, **k):
    captured["frames"] = np.asarray(frames)
    return _orig(path, frames, *a, **k)
svs.imageio.mimwrite = grab
CFG = {"jobs": [{"name": "sdragon_el+0", "hero": "hero_sdragon.png", "azimuths": [17.142857142857142, 34.285714285714285, 51.42857142857143, 68.57142857142857, 85.71428571428571, 102.85714285714286, 120.0, 137.14285714285714, 154.28571428571428, 171.42857142857142, 188.57142857142856, 205.71428571428572, 222.85714285714286, 240.0, 257.1428571428571, 274.2857142857143, 291.42857142857144, 308.57142857142856, 325.7142857142857, 342.85714285714283, 0.0], "elevations": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}, {"name": "gptdragon_el+0", "hero": "hero_gptdragon.png", "azimuths": [17.142857142857142, 34.285714285714285, 51.42857142857143, 68.57142857142857, 85.71428571428571, 102.85714285714286, 120.0, 137.14285714285714, 154.28571428571428, 171.42857142857142, 188.57142857142856, 205.71428571428572, 222.85714285714286, 240.0, 257.1428571428571, 274.2857142857143, 291.42857142857144, 308.57142857142856, 325.7142857142857, 342.85714285714283, 0.0], "elevations": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}], "steps": 25, "decoding_t": 2, "budget_min": 500}
from PIL import Image
out_root = "/kaggle/working/out"
W, NW = int(os.environ["SV3D_WORKER"]), int(os.environ["SV3D_NWORK"])
for job in CFG["jobs"][W::NW]:
    if time.time() - t_start > CFG.get("budget_min", 520) * 60:
        print("time budget reached, skipping", job["name"]); continue
    hero = locate(job["hero"])
    t0 = time.time()
    captured.clear()
    svs.sample(input_path=hero, version="sv3d_p", elevations_deg=job["elevations"], azimuths_deg=job["azimuths"],
               decoding_t=CFG.get("decoding_t", 2), num_steps=CFG.get("steps", 50), seed=job.get("seed", 23),
               output_folder=f"/kaggle/working/tmp/{job['name']}", device="cuda")
    frames = captured["frames"]
    od = os.path.join(out_root, job["name"]); os.makedirs(od, exist_ok=True)
    for f, az, el in zip(frames, job["azimuths"], job["elevations"]):
        Image.fromarray(f).save(os.path.join(od, f"e{int(round(el)):+03d}_a{int(round(az)) % 360:03d}.png"))
    json.dump({"azimuths": job["azimuths"], "elevations": job["elevations"], "seconds": time.time() - t0,
               "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9}, open(os.path.join(od, "poses.json"), "w"))
    print(job["name"], "done", round(time.time() - t0), "s, peak", round(torch.cuda.max_memory_allocated() / 1e9, 1), "GB", flush=True)
    torch.cuda.reset_peak_memory_stats()

