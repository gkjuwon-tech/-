"""Codex CLI image_generation wrapper: run a prompt with attached images, collect the newest output."""
import glob
import os
import shutil
import subprocess
import time

GEN_DIR = os.path.expanduser("~/.codex/generated_images")


def generate(prompt, images, out_path, workdir=None, timeout=600, retries=2):
    workdir = workdir or os.path.dirname(os.path.abspath(out_path))
    full = (prompt.strip() + "\n\nUse your image generation tool exactly once. "
            "Do not write code, do not post-process with scripts. Just generate the image.")
    for attempt in range(retries + 1):
        start = time.time()
        cmd = ["codex", "exec", "--skip-git-repo-check", "--sandbox", "workspace-write"]
        if images:
            cmd.append("--image=" + ",".join(os.path.abspath(p) for p in images))
        cmd.append("-")
        subprocess.run(cmd, input=full, text=True, cwd=workdir, timeout=timeout,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        new = [p for p in glob.glob(os.path.join(GEN_DIR, "**", "*.png"), recursive=True)
               if os.path.getmtime(p) >= start - 1]
        if new:
            shutil.copy(max(new, key=os.path.getmtime), out_path)
            return out_path
        print(f"  [imagegen] no image produced (attempt {attempt + 1}), retrying")
    raise RuntimeError("image generation failed: " + out_path)
