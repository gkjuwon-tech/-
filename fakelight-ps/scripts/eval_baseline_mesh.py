# 기준선 메쉬 채점: 3D 생성 모델(TripoSR 등) 결과를 정답 버니에 맞춰 정렬(24방향 x ICP)하고 Chamfer/F-score와 5방향 비교 렌더를 만든다.
# 우리 파이프라인에는 3D 생성 모델을 쓰지 않는다. 이건 이겨야 할 상대의 점수를 재는 용도.
import sys, itertools, numpy as np, trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from PIL import Image
sys.path.insert(0, 'scripts')
from render_front import look_rotation, rasterize, shade

def norm_mesh(m):
    v = m.vertices - m.bounding_box.centroid
    return v / np.linalg.norm(m.extents), m.faces

gt = trimesh.load('assets/stanford-bunny.obj', force='mesh', process=True)
gv, gf = norm_mesh(gt)
gv = gv @ look_rotation(80).T                                   # 입력 사진과 같은 프레임 (정면 = +z 카메라)
tm = trimesh.load('/home/user/triposr/out/0/mesh.obj', force='mesh', process=True)
tv, tf = norm_mesh(tm)

gs = trimesh.Trimesh(gv, gf).sample(20000); ts = trimesh.Trimesh(tv, tf).sample(20000)
tree_g = cKDTree(gs)
# 정렬: 24개 축 회전 × ICP, 비용 최소 (TripoSR 좌표계를 모르니 전수 탐색)
cands = Rot.create_group('O').as_matrix()
best = None
for R in cands:
    M = np.eye(4); M[:3, :3] = R
    mtx, _, cost = trimesh.registration.icp(ts, gs, initial=M, max_iterations=40, scale=True)
    if best is None or cost < best[1]:
        best = (mtx, cost)
mtx = best[0]
tv_al = trimesh.transform_points(tv, mtx)
ts_al = trimesh.transform_points(ts, mtx)
d_ts = cKDTree(gs).query(ts_al)[0]; d_st = cKDTree(ts_al).query(gs)[0]
diag = 1.0  # 정답은 대각선 1로 정규화
res = {'chamfer': float((d_ts.mean() + d_st.mean()) / 2)}
for t in [0.01, 0.02]:
    p = (d_ts < t).mean(); r = (d_st < t).mean(); res[f'F@{int(t*100)}%'] = float(2 * p * r / (p + r + 1e-12)); res[f'P@{int(t*100)}%']=float(p); res[f'R@{int(t*100)}%']=float(r)
print({k: round(v, 4) for k, v in res.items()})

def render(v, f, yaw, pitch=0, res_=384):
    R = look_rotation(yaw)
    if pitch:
        c, s = np.cos(np.radians(pitch)), np.sin(np.radians(pitch)); R = np.array([[1,0,0],[0,c,-s],[0,s,c]]) @ R
    vv = v @ R.T * 1.6
    m = trimesh.Trimesh(vv, f, process=False)
    n, mask = rasterize(vv, f, m.vertex_normals, res_)
    img = shade(n, mask); img[~mask] = 1.0
    return Image.fromarray((img * 255).astype(np.uint8))

tmesh = trimesh.Trimesh(tv_al, tf[:, ::-1]).simplify_quadric_decimation(face_count=30000) if hasattr(trimesh.Trimesh, 'simplify_quadric_decimation') else trimesh.Trimesh(tv_al, tf)
views = [(0, 0, '정면(입력)'), (90, 0, '옆'), (180, 0, '뒤'), (270, 0, '반대 옆'), (0, 60, '위')]
S = 384
sheet = Image.new('RGB', (S * len(views), S * 2), 'white')
for i, (y, p, _) in enumerate(views):
    sheet.paste(render(gv, gf, y, p), (i * S, 0))
    sheet.paste(render(tmesh.vertices, tmesh.faces, y, p), (i * S, S))
sheet.save('results/triposr_bunny/sheet.png')
import json; json.dump(res, open('results/triposr_bunny/scores.json', 'w'), indent=2)
