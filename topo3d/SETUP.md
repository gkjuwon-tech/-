# topo3d 환경

> 임시로 꼬질룡 레포 안에 둔 폴더예요. 별도 `topo3d` 레포가 생기면 옮길 예정.
> `.venv`, `models/`, `da3/`는 용량 때문에 커밋하지 않아요. 아래 재구축 절차로 이 폴더 안에 다시 만들면 돼요.

- Python 3.11 venv: `.venv` (`. .venv/bin/activate`)
- PyTorch 2.14.0 CPU (GPU 없음 · 4코어 · RAM 15GB)
- Depth Anything 3: `da3/` (editable 설치, commit 3d835ec)
- 가중치: `models/DA3-BASE` (517MB) → `DepthAnything3.from_pretrained('models/DA3-BASE')`
- xformers 없음 (CPU라 불필요, DA3가 자동 우회)
- 시스템 라이브러리 (open3d용): libegl1 libgl1 libgomp1 libusb-1.0-0
- 누락된 의존성 추가 설치: addict, pycolmap, evo

## 재구축 (topo3d/ 안에서)
```
apt-get install -y libegl1 libgl1 libgomp1 libusb-1.0-0
python3 -m venv .venv && . .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.lock.txt
git clone https://github.com/ByteDance-Seed/Depth-Anything-3 da3 && pip install --no-deps -e da3
```

## 연기 테스트 결과
예제 2장, 280×504, CPU 추론 8.2초 → depth/conf/extrinsics/intrinsics 정상 출력

## 가중치 받기
```
python -c "from huggingface_hub import snapshot_download; snapshot_download('depth-anything/DA3-BASE', local_dir='models/DA3-BASE')"
```

## 테스트 데이터: 스탠포드 루시 8뷰
`inputs/lucy_8v/`에 커밋됨: RGBA·회색배경 PNG 8뷰(1024px), `cameras.json`(OpenCV K, R, t), `sheet.png`.
정답 깊이/법선(`*_depth.npy`, `*_normal.npy`, 133MB)은 용량 때문에 커밋하지 않음. 재생성 방법:
```
mkdir -p data/lucy && cd data/lucy
curl -O http://graphics.stanford.edu/data/3Dscanrep/lucy.tar.gz && tar xzf lucy.tar.gz && cd ../..
python tools/prep_lucy.py data/lucy/lucy.ply data/lucy/lucy_4m.ply
python tools/render_turnaround.py data/lucy/lucy_4m.ply data/lucy/turnaround_8v --size 1024 --ssaa 2 --yaw 180
```
