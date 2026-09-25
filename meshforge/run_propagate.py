import sys, json, numpy as np
from PIL import Image
from skimage import measure
from mf.hull import Grid
from mf.propagate import propagate

pre = sys.argv[1]
d = np.load(pre + ".npz"); occ = d["occ"]; off = {int(k): v for k, v in json.loads(str(d["off"])).items()}
g = Grid(step=int(d["step"]))
rgb = {t: np.asarray(Image.open(f"{pre}_canon_{t}.png")) for t in off}
# mesh export
verts, faces, _, _ = measure.marching_cubes(np.pad(occ, 1).astype(np.float32), 0.5)
verts = (verts - 1) * g.step
verts[:, 0] += g.xs[0]; verts[:, 2] += g.xs[0]
import trimesh
mesh = trimesh.Trimesh(verts[:, [0, 1, 2]] * [1, -1, 1], faces[:, ::-1])
mesh.export(pre + "_hull.obj")
print("mesh", mesh.vertices.shape, mesh.faces.shape, "watertight", mesh.is_watertight)
# propagate front -> 30 and compare with GPT's own 30 view
for tgt in [30, 90]:
    guide, filled, sil = propagate(g, occ, off, {0: rgb[0]}, tgt)
    real = rgb[tgt]
    bg = np.full_like(guide, 128); canvas = np.where(filled[..., None], guide, bg)
    canvas[sil & ~filled] = (255, 0, 255)  # unseen -> to be generated
    Image.fromarray(np.concatenate([real, canvas], 1)).save(f"{pre}_prop_0to{tgt}.png")
    print(tgt, "filled fraction of silhouette", round(filled.sum() / sil.sum(), 3))
