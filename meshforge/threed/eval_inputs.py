"""Input-level scores: normal error vs the ground-truth normals seen by the true cameras,
and camera error vs the degradation truth. usage: python eval_inputs.py DATA/NAME [DATA/NAME_fixed ...]"""
import json, os, sys
import numpy as np
from PIL import Image

for d in sys.argv[1:]:
    names = list(json.load(open(os.path.join(d, "views", "cameras.json")))["views"])
    gtd = os.path.join(d, "normals_gt") if os.path.isdir(os.path.join(d, "normals_gt")) else os.path.join(d.replace("_fixed", ""), "normals_gt")
    errs = []
    for v in names:
        n = np.load(os.path.join(d, "normals", f"{v}.npy")); g = np.load(os.path.join(gtd, f"{v}.npy"))
        m = (np.linalg.norm(n, axis=-1) > 0.5) & (np.linalg.norm(g, axis=-1) > 0.5)
        e = np.degrees(np.arccos(np.clip((n * g).sum(-1)[m], -1, 1)))
        errs.append((v, e.mean(), np.median(e)))
    print(f"{os.path.basename(d):<22} normal error mean {np.mean([e[1] for e in errs]):5.1f}  median {np.mean([e[2] for e in errs]):5.1f} deg  |",
          " ".join(f"{v}:{m:.0f}" for v, m, _ in errs))
