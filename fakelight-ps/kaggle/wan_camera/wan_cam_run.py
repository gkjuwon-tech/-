# Wan2.2-Fun-5B-Control-Camera 실행기: 공식 predict_v2v_control_camera_5b.py의 설정 줄만 바꿔서 실행한다.
import re, subprocess, sys
tag, img, pose, prompt, size, mem = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
src = open("examples/wan2.2_fun/predict_v2v_control_camera_5b.py").read()
h, w = size.split("x")
subs = {
    r'^GPU_memory_mode\s*=.*$': f'GPU_memory_mode     = "{mem}"',
    r'^sample_size\s*=.*$': f'sample_size         = [{h}, {w}]',
    r'^control_camera_txt\s*=.*$': f'control_camera_txt      = "{pose}"',
    r'^start_image\s*=.*$': f'start_image             = "{img}"',
    r'^prompt\s*=.*$': 'prompt                  = ' + repr(prompt),
    r'^save_path\s*=.*$': f'save_path               = "samples/{tag}"',
}
for pat, rep in subs.items():
    src, n = re.subn(pat, rep, src, count=1, flags=re.M)
    assert n == 1, pat
open(f"run_{tag}.py", "w").write(src)
sys.exit(subprocess.call([sys.executable, f"run_{tag}.py"]))
