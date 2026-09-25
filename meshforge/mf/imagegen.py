"""Codex CLI image_generation wrapper. Safe to call from parallel threads: the output image
is located through the session (thread) id that `codex exec --json` reports."""
import glob
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

GEN_DIR = os.path.expanduser("~/.codex/generated_images")
SUFFIX = ("\n\nUse your image generation tool exactly once. Do not write code, "
          "do not post-process with scripts. Just generate the image.")


def generate(prompt, images, out_path, workdir=None, timeout=900, retries=2):
    workdir = workdir or os.path.dirname(os.path.abspath(out_path))
    for attempt in range(retries + 1):
        cmd = ["codex", "exec", "--json", "--skip-git-repo-check", "--sandbox", "workspace-write"]
        if images:
            cmd.append("--image=" + ",".join(os.path.abspath(p) for p in images))
        cmd.append("-")
        try:
            r = subprocess.run(cmd, input=prompt.strip() + SUFFIX, text=True, cwd=workdir,
                               timeout=timeout, capture_output=True)
            out = r.stdout
        except subprocess.TimeoutExpired:
            out = ""
        paths = re.findall(r"generated_images/[^\"\s']+?\.png", out)
        thread = None
        for line in out.splitlines():
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get("type") == "thread.started":
                thread = ev.get("thread_id")
        cands = [os.path.join(os.path.expanduser("~/.codex"), p) for p in paths]
        if thread:
            cands += glob.glob(os.path.join(GEN_DIR, thread, "*.png"))
        cands = [p for p in cands if os.path.exists(p)]
        if cands:
            shutil.copy(max(cands, key=os.path.getmtime), out_path)
            return out_path
        print(f"  [imagegen] no image for {os.path.basename(out_path)} (attempt {attempt + 1})", flush=True)
    raise RuntimeError("image generation failed: " + out_path)


def generate_many(jobs, workers=4):
    """jobs: list of (prompt, images, out_path). Returns list of paths (None on failure)."""
    def run(job):
        try:
            return generate(*job)
        except Exception as e:  # keep the batch going
            print("  [imagegen]", e, flush=True)
            return None
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(run, jobs))
