"""Corrector: turn realistic (bad) stage2 inputs into clean ones, using only physics the
images must obey. Reads data/<name>/{views,normals}, writes data/<name>_fixed/ in the same format.

1. cameras   per view, the small rotation about the object and in-plane shift that make its
             silhouette agree with the hull carved by all the OTHER views (leave-one-out IoU)
2. masks     an orthographic silhouette equals the mirrored silhouette of the opposite view;
             opposite pairs are averaged, their independent raggedness cancels
3. normals   (a) along the silhouette the true normal is known exactly: it lies in the image
             plane, perpendicular to the outline. The estimator's error there is measured as a
             rotation per rim pixel and spread smoothly inward (normalised convolution), then
             undone -- this removes the smooth 'bent surface' bias of monocular estimators.
             (b) a single global rotation per view (Kabsch on rim pixels) goes first.

usage: python corrector.py DATA_DIR/NAME [--out DATA_DIR/NAME_fixed]"""
import argparse, json, os, shutil, time
import numpy as np
from PIL import Image
from scipy import ndimage

log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- cameras
def axes(M):
    M = np.asarray(M, float)
    return M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3]


def look_center(meta):
    A, b = np.zeros((3, 3)), np.zeros(3)
    for v in meta["views"].values():
        _, _, back, loc = axes(v["matrix_world"])
        P = np.eye(3) - np.outer(back, back)
        A += P; b += P @ loc
    return np.linalg.solve(A, b)


def rot(axis, deg):
    axis = axis / np.linalg.norm(axis); t = np.radians(deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def perturb(M, center, daz, delv, du, dv, px):
    right, up, back, loc = axes(M)
    R = rot(np.array([0, 0, 1.0]), daz) @ rot(right, -delv)   # +delv raises the camera
    r2, u2, b2 = R @ right, R @ up, R @ back
    l2 = center + R @ (loc - center) - du * px * r2 + dv * px * u2
    N = np.eye(4); N[:3, 0], N[:3, 1], N[:3, 2], N[:3, 3] = r2, u2, b2, l2
    return N


def project(M, P, ortho, res):
    right, up, back, loc = axes(M)
    rel = P - loc
    return (0.5 - rel @ up / ortho) * res - 0.5, (rel @ right / ortho + 0.5) * res - 0.5


def sample(mask, r, c):
    ri, ci = np.round(r).astype(int), np.round(c).astype(int)
    ok = (ri >= 0) & (ri < mask.shape[0]) & (ci >= 0) & (ci < mask.shape[1])
    out = np.zeros(r.shape, bool); out[ok] = mask[ri[ok], ci[ok]]
    return out


def splat(r, c, res, grow=1):
    img = np.zeros((res, res), bool)
    ri, ci = np.round(r).astype(int), np.round(c).astype(int)
    ok = (ri >= 0) & (ri < res) & (ci >= 0) & (ci < res)
    img[ri[ok], ci[ok]] = True
    return ndimage.binary_closing(ndimage.binary_dilation(img, iterations=grow), iterations=2)


def iou(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else 1.0


def calibrate(meta, masks, center, lo_res=256, n=176, rounds=3, prior=0.004):
    """Coordinate descent per view. Objective: the view's silhouette must lie inside the hull
    carved by the other views (that hull is a superset of the object, so containment -- not
    IoU, which a fat hull biases -- is the right test), with a mild preference for small fixes."""
    ortho = meta["ortho_scale"]; R = meta["resolution"][0]
    small = {v: np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((lo_res, lo_res), Image.BILINEAR)) > 127
             for v, m in masks.items()}
    g = (np.arange(n) + 0.5) / n - 0.5
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    P = center + ortho * np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    Ms = {v: np.array(meta["views"][v]["matrix_world"]) for v in masks}
    params = {v: np.zeros(4) for v in masks}
    px_full = ortho / R
    cur = lambda v: perturb(Ms[v], center, *params[v], px_full)
    steps = np.array([2.0, 2.0, 3.0, 3.0])            # deg, deg, px, px (full resolution)
    limit = np.array([8.0, 6.0, 10.0, 10.0])
    for rd in range(rounds):
        scores = {}
        for v in masks:
            loo = np.ones(len(P), bool)
            for w in masks:
                if w != v:
                    loo &= sample(small[w], *project(cur(w), P, ortho, lo_res))
            Q = P[loo]
            mv = small[v]
            def score(p):
                M = perturb(Ms[v], center, *p, px_full)
                proj = splat(*project(M, Q, ortho, lo_res), lo_res)
                contain = (mv & proj).sum() / max(mv.sum(), 1)
                return contain - prior * np.abs(p / steps).sum()
            best = score(params[v])
            step = steps / (2 ** rd)
            for _ in range(4):
                improved = False
                for k in range(4):
                    for sgn in (-1, 1):
                        p = params[v].copy(); p[k] = np.clip(p[k] + sgn * step[k], -limit[k], limit[k])
                        s_ = score(p)
                        if s_ > best + 1e-5:
                            best, params[v], improved = s_, p, True
                if not improved:
                    break
            scores[v] = best
        log(f"calibration round {rd}: containment", {v: round(float(s_), 4) for v, s_ in scores.items()})
    return {v: cur(v) for v in masks}, {v: p.tolist() for v, p in params.items()}


# ---------------------------------------------------------------- masks
def opposite_pairs(Ms):
    names = list(Ms); pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if np.dot(axes(Ms[a])[2], axes(Ms[b])[2]) < -0.995:
                pairs.append((a, b))
    return pairs


def mirror_consensus(masks, Ms, meta, center):
    """Mirror view b into view a's image through its actual camera and average."""
    out = {v: m.copy() for v, m in masks.items()}
    ortho, R = meta["ortho_scale"], meta["resolution"][0]
    for a, b in opposite_pairs(Ms):
        ra, ua, ba, la = axes(Ms[a]); rb, ub, bb, lb = axes(Ms[b])
        # a pixel (row, col) of a sees the ray P(t) = la + x ra + y ua + t ba; its image in b
        rr, cc = np.mgrid[0:R, 0:R].astype(np.float64)
        x = ((cc + 0.5) / R - 0.5) * ortho; y = (0.5 - (rr + 0.5) / R) * ortho
        P = la + x[..., None] * ra + y[..., None] * ua
        relb = P - lb
        cb = (relb @ rb / ortho + 0.5) * R - 0.5; rb_ = (0.5 - relb @ ub / ortho) * R - 0.5
        mb_in_a = ndimage.map_coordinates(masks[b], [rb_, cb], order=1, cval=0)
        avg = 0.5 * (masks[a] + mb_in_a)
        out[a] = avg
        # and back into b
        rr2, cc2 = rr, cc
        x2 = ((cc2 + 0.5) / R - 0.5) * ortho; y2 = (0.5 - (rr2 + 0.5) / R) * ortho
        P2 = lb + x2[..., None] * rb + y2[..., None] * ub
        rela = P2 - la
        ca = (rela @ ra / ortho + 0.5) * R - 0.5; ra_ = (0.5 - rela @ ua / ortho) * R - 0.5
        out[b] = ndimage.map_coordinates(avg, [ra_, ca], order=1, cval=0)
        log(f"mirror pair {a}|{b}: IoU before {iou(masks[a] > .5, mb_in_a > .5):.3f}")
    return out


# ---------------------------------------------------------------- normals
def rim_targets(mask, band=2):
    """Pixels just inside the outline and the exact normal there (image plane, outward)."""
    m = mask > 0.5
    sm = ndimage.gaussian_filter(m.astype(np.float32), 2.0)
    gy, gx = np.gradient(sm)                     # points inward (toward inside)
    rim = m & ~ndimage.binary_erosion(m, iterations=band)
    nx, ny = -gx, gy                              # outward; image rows go down, camera y goes up
    L = np.hypot(nx, ny)
    ok = rim & (L > 1e-3)
    t = np.zeros(mask.shape + (3,), np.float32)
    t[ok, 0], t[ok, 1] = nx[ok] / L[ok], ny[ok] / L[ok]
    return ok, t


def min_rotation(a, b):
    """Axis-angle vectors rotating unit vectors a onto b."""
    ax = np.cross(a, b)
    s = np.linalg.norm(ax, axis=-1, keepdims=True)
    c = (a * b).sum(-1, keepdims=True)
    ang = np.arctan2(s, c)
    return ax / np.maximum(s, 1e-9) * ang


def apply_rotation(n, aa):
    th = np.linalg.norm(aa, axis=-1, keepdims=True)
    k = aa / np.maximum(th, 1e-9)
    c, s = np.cos(th), np.sin(th)
    return n * c + np.cross(k, n) * s + k * (k * n).sum(-1, keepdims=True) * (1 - c)


def kabsch(A, B):
    H = A.T @ B
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


def correct_normals(n, mask, sigma_frac=0.06, max_deg=40):
    m = mask > 0.5
    ok, t = rim_targets(mask)
    ok &= np.linalg.norm(n, axis=-1) > 0.5
    src = n[ok] / np.maximum(np.linalg.norm(n[ok], axis=1, keepdims=True), 1e-9)
    # rim normals from an estimator sit a pixel or two inside: allow them some tilt toward the camera
    tgt = t[ok].copy()
    tgt[:, 2] = np.clip(src[:, 2], 0, 0.35)
    tgt /= np.linalg.norm(tgt, axis=1, keepdims=True)
    Rg = kabsch(src, tgt)
    n1 = n @ Rg.T
    src1 = n1[ok] / np.maximum(np.linalg.norm(n1[ok], axis=1, keepdims=True), 1e-9)
    aa = min_rotation(src1, tgt)
    ang = np.degrees(np.linalg.norm(aa, axis=1))
    keep = ang < max_deg                          # outliers: thin parts, bad rim estimates
    field = np.zeros(mask.shape + (3,), np.float32); wgt = np.zeros(mask.shape, np.float32)
    rr, cc = np.nonzero(ok)
    field[rr[keep], cc[keep]] = aa[keep]; wgt[rr[keep], cc[keep]] = 1.0
    sigma = sigma_frac * mask.shape[0]
    num = np.stack([ndimage.gaussian_filter(field[..., k], sigma) for k in range(3)], -1)
    den = ndimage.gaussian_filter(wgt, sigma)
    corr = num / np.maximum(den[..., None], 1e-6)
    # fade the correction out where no rim is near (don't invent it deep inside)
    conf = np.clip(den / (den[m].max() * 0.15 + 1e-9), 0, 1)[..., None]
    n2 = apply_rotation(n1, corr * conf)
    n2 /= np.maximum(np.linalg.norm(n2, axis=-1, keepdims=True), 1e-9)
    n2[~m] = 0
    n2[m & (n2[..., 2] < 0.0), 2] = 0.0
    return n2.astype(np.float32), float(np.degrees(np.arccos(np.clip((np.trace(Rg) - 1) / 2, -1, 1)))), float(np.median(ang))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("--out", default=None)
    ap.add_argument("--no-cameras", action="store_true"); ap.add_argument("--no-masks", action="store_true")
    ap.add_argument("--no-normals", action="store_true")
    a = ap.parse_args()
    out = a.out or a.src.rstrip("/") + "_fixed"
    meta = json.load(open(os.path.join(a.src, "views", "cameras.json")))
    names = list(meta["views"])
    masks = {v: np.asarray(Image.open(os.path.join(a.src, "views", "mask", f"{v}.png")).convert("L"), np.float32) / 255 for v in names}
    normals = {v: np.load(os.path.join(a.src, "normals", f"{v}.npy")).astype(np.float32) for v in names}
    center = look_center(meta)
    Ms = {v: np.array(meta["views"][v]["matrix_world"]) for v in names}
    if not a.no_cameras:
        Ms, params = calibrate(meta, masks, center)
        log("camera corrections (d_az, d_el, du, dv):", {v: [round(x, 2) for x in p] for v, p in params.items()})
    if not a.no_masks:
        masks = mirror_consensus(masks, Ms, meta, center)
    if not a.no_normals:
        for v in names:
            normals[v], g_deg, rim_med = correct_normals(normals[v], masks[v])
            log(f"normals {v}: global rotation {g_deg:.1f} deg, rim error median {rim_med:.1f} deg (after global)")
    shutil.rmtree(out, ignore_errors=True)
    for d in ("views/mask", "views/rgb", "normals"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    for v in names:
        Image.fromarray((np.clip(masks[v], 0, 1) * 255).round().astype(np.uint8)).save(os.path.join(out, "views", "mask", f"{v}.png"))
        shutil.copy(os.path.join(a.src, "views", "rgb", f"{v}.png"), os.path.join(out, "views", "rgb", f"{v}.png"))
        np.save(os.path.join(out, "normals", f"{v}.npy"), normals[v] * (masks[v][..., None] > 0.5))
        meta["views"][v]["matrix_world"] = Ms[v].tolist()
    json.dump(meta, open(os.path.join(out, "views", "cameras.json"), "w"), indent=1)
    for f in ("gt_mesh.ply", "degradation_truth.json"):
        if os.path.exists(os.path.join(a.src, f)):
            shutil.copy(os.path.join(a.src, f), os.path.join(out, f))
    log("wrote", out)


if __name__ == "__main__":
    main()
