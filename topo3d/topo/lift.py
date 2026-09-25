"""깊이맵 → 쿼드 메시 (픽셀 1개 = 쿼드 1개).

꼭짓점은 픽셀 모서리 격자 (H+1)x(W+1)에 놓인다. 그래서 쿼드를 같은 카메라로 투영하면
마스크 픽셀과 정확히 같은 영역을 덮는다.
이웃 픽셀 사이 깊이가 끊기면(discontinuity) 그 모서리의 꼭짓점을 공유하지 않고 쪼갠다.
뿔 사이, 손가락 사이, 날개와 몸통 사이를 이어 붙이지 않기 위한 규칙.
"""
import numpy as np
from numba import njit


def discontinuity_edges(depth, mask, K, max_slope_deg=84.0, rel_jump=0.02):
    """가로·세로 이웃 픽셀 쌍이 끊겼는지 판정.

    두 조건 중 하나면 끊긴 것으로 본다.
    - 표면 기울기가 시선 기준 max_slope_deg를 넘음 (거의 시선과 평행한 벽 = 실제로는 절벽)
    - 깊이 차가 rel_jump * 깊이 보다 큼
    반환: cut_h (H, W-1), cut_v (H-1, W) bool.
    """
    f = K[0, 0]
    tan_max = np.tan(np.deg2rad(max_slope_deg))

    def cut(d0, d1, m0, m1):
        dz = np.abs(d0 - d1)
        z = np.minimum(d0, d1)
        pix = z / f                           # 픽셀 한 칸의 월드 폭
        return m0 & m1 & ((dz > tan_max * pix) | (dz > rel_jump * z))

    cut_h = cut(depth[:, :-1], depth[:, 1:], mask[:, :-1], mask[:, 1:])
    cut_v = cut(depth[:-1, :], depth[1:, :], mask[:-1, :], mask[1:, :])
    return cut_h, cut_v


@njit(cache=True)
def _find(p, a):
    while p[a] != a:
        p[a] = p[p[a]]
        a = p[a]
    return a


@njit(cache=True)
def _build(mask, depth, cut_h, cut_v):
    """픽셀별 4모서리 꼭짓점 번호 (H,W,4: TL,TR,BR,BL)와 꼭짓점 깊이를 만든다."""
    H, W = mask.shape
    pix_corner = np.full((H, W, 4), -1, np.int64)
    vdepth = np.zeros(H * W * 4, np.float64)
    vcx = np.zeros(H * W * 4, np.int64)
    vcy = np.zeros(H * W * 4, np.int64)
    nv = 0
    # 모서리 (ci, cj)를 둘러싼 픽셀: 0=(ci-1,cj-1) 1=(ci-1,cj) 2=(ci,cj-1) 3=(ci,cj)
    # 각 픽셀에서 이 모서리의 역할: 0→BR(2) 1→BL(3) 2→TR(1) 3→TL(0)
    role = np.array([2, 3, 1, 0])
    par = np.zeros(4, np.int64)
    for ci in range(H + 1):
        for cj in range(W + 1):
            pi = np.array([ci - 1, ci - 1, ci, ci])
            pj = np.array([cj - 1, cj, cj - 1, cj])
            inside = np.zeros(4, np.bool_)
            any_in = False
            for k in range(4):
                if 0 <= pi[k] < H and 0 <= pj[k] < W and mask[pi[k], pj[k]]:
                    inside[k] = True
                    any_in = True
            if not any_in:
                continue
            for k in range(4):
                par[k] = k
            # 연결: 0-1 (가로, 위 줄), 2-3 (가로, 아래 줄), 0-2 (세로, 왼쪽), 1-3 (세로, 오른쪽)
            if inside[0] and inside[1] and not cut_h[ci - 1, cj - 1]:
                par[_find(par, 1)] = _find(par, 0)
            if inside[2] and inside[3] and not cut_h[ci, cj - 1]:
                par[_find(par, 3)] = _find(par, 2)
            if inside[0] and inside[2] and not cut_v[ci - 1, cj - 1]:
                par[_find(par, 2)] = _find(par, 0)
            if inside[1] and inside[3] and not cut_v[ci - 1, cj]:
                par[_find(par, 3)] = _find(par, 1)
            gid = np.full(4, -1, np.int64)
            gsum = np.zeros(4)
            gcnt = np.zeros(4)
            for k in range(4):
                if inside[k]:
                    r = _find(par, k)
                    gsum[r] += depth[pi[k], pj[k]]
                    gcnt[r] += 1
            for k in range(4):
                if inside[k]:
                    r = _find(par, k)
                    if gid[r] < 0:
                        gid[r] = nv
                        vdepth[nv] = gsum[r] / gcnt[r]
                        vcx[nv] = cj
                        vcy[nv] = ci
                        nv += 1
                    pix_corner[pi[k], pj[k], role[k]] = gid[r]
    return pix_corner, vdepth[:nv], vcx[:nv], vcy[:nv]


def lift(mask, depth, K, R, t, cut_h=None, cut_v=None):
    """마스크 픽셀마다 쿼드 하나. 반환 dict(V 월드좌표, Q 쿼드(카메라를 향하는 감김), pix 쿼드의 픽셀 (y,x))."""
    mask = mask.astype(bool)
    if cut_h is None:
        cut_h, cut_v = discontinuity_edges(depth, mask, K)
    pc, vd, vcx, vcy = _build(mask, depth.astype(np.float64), cut_h, cut_v)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    Xc = np.stack([(vcx - cx) / fx * vd, (vcy - cy) / fy * vd, vd], 1)
    R = np.asarray(R)
    V = (Xc - np.asarray(t)) @ R            # camera → world (R 직교)
    ys, xs = np.nonzero(mask)
    c = pc[ys, xs]                          # TL, TR, BR, BL
    Q = c[:, [0, 3, 2, 1]]                  # TL, BL, BR, TR → 카메라 쪽 법선
    return {"V": V, "Q": Q, "pix": np.stack([ys, xs], 1), "cut_h": cut_h, "cut_v": cut_v}
