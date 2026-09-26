"""모델별 mv_grid.png를 세로로 쌓고 왼쪽에 모델 이름을 붙여 눈 비교용 한 장을 만든다.

사용법:
    python scripts/compare_grids.py --out results/audition_compare.png \
        results/mvadapter_v2/views results/audition/stable-diffusion-xl-base-1.0 ...
"""
import argparse
import json
import os

from PIL import Image, ImageDraw, ImageFont

LABEL_W = 360


def label_for(folder):
    stats = os.path.join(folder, "stats.json")
    if os.path.exists(stats):
        s = json.load(open(stats))
        name = s.get("base_model", os.path.basename(folder)).split("/")[-1]
        extra = f"{s['gen_seconds']}s  {s['peak_mem_gb']}GB" if "gen_seconds" in s else s.get("error", "")
        return name, extra
    return os.path.basename(folder), ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--row-height", type=int, default=256)
    args = ap.parse_args()

    font = ImageFont.load_default(size=26)
    small = ImageFont.load_default(size=20)
    rows = []
    for folder in args.folders:
        grid_path = os.path.join(folder, "mv_grid.png")
        if not os.path.exists(grid_path):
            print(f"skip (no grid): {folder}")
            continue
        grid = Image.open(grid_path).convert("RGB")
        scale = args.row_height / grid.height
        grid = grid.resize((round(grid.width * scale), args.row_height))
        row = Image.new("RGB", (LABEL_W + grid.width, args.row_height), "white")
        row.paste(grid, (LABEL_W, 0))
        name, extra = label_for(folder)
        d = ImageDraw.Draw(row)
        d.text((16, args.row_height // 2 - 30), name, fill="black", font=font)
        d.text((16, args.row_height // 2 + 8), extra, fill="gray", font=small)
        rows.append(row)

    width = max(r.width for r in rows)
    canvas = Image.new("RGB", (width, sum(r.height for r in rows)), "white")
    y = 0
    for r in rows:
        canvas.paste(r, (0, y))
        y += r.height
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    canvas.save(args.out)
    print(f"saved {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
