"""스탠포드 루시 원본(2,800만 면)을 렌더링용으로 준비한다.

400만 면으로 감량 → 바닥 z=0, xy 중심 정렬 → 키 1로 정규화 → Z-up을 Y-up (x, z, -y)으로 변환.
사용: python tools/prep_lucy.py data/lucy/lucy.ply data/lucy/lucy_4m.ply
"""
import sys

import fast_simplification
import numpy as np
import trimesh

src, dst = sys.argv[1], sys.argv[2]
m = trimesh.load(src, process=False)
v, f = fast_simplification.simplify(m.vertices.astype(np.float32), m.faces.astype(np.int32),
                                    target_count=4_000_000)
c = (v.min(0) + v.max(0)) / 2
v = v - np.array([c[0], c[1], v[:, 2].min()])
v = v / v[:, 2].max()
v = np.stack([v[:, 0], v[:, 2], -v[:, 1]], 1)
trimesh.Trimesh(v, f, process=False).export(dst)
print(dst, v.shape, f.shape)
