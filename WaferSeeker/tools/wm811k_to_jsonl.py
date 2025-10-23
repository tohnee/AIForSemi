import os
import json
import argparse
import numpy as np
from PIL import Image, ImageOps

# WM811K 常见标签到缺陷类型映射（可根据需要扩展）
WM811K_TO_DEFECT = {
    'Scratch': 'SCRATCH',
    'Center': 'PARTICLE',
    'Donut': 'PARTICLE',
    'Edge-Loc': 'PARTICLE',
    'Edge-Ring': 'PARTICLE',
    'Loc': 'PARTICLE',
    'Near-full': 'DISCOLOR',
    'Random': 'PARTICLE',
    'None': 'NONE',
}


def connected_components(binary_map):
    H, W = binary_map.shape
    visited = np.zeros_like(binary_map, dtype=bool)
    components = []
    for y in range(H):
        for x in range(W):
            if binary_map[y, x] and not visited[y, x]:
                stack = [(y, x)]
                visited[y, x] = True
                ys, xs = [y], [x]
                while stack:
                    cy, cx = stack.pop()
                    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < H and 0 <= nx < W and binary_map[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
                            ys.append(ny)
                            xs.append(nx)
                y1, y2 = min(ys), max(ys)
                x1, x2 = min(xs), max(xs)
                components.append((x1, y1, x2+1, y2+1))
    return components


def render_wafer_map(array, out_size=512):
    # 输入 array: 2D，值域 {0,1} 或更多类别
    arr = (array > 0).astype(np.uint8) * 255
    img = Image.fromarray(arr, mode='L')
    img = ImageOps.expand(img, border=8, fill=0)
    img = img.resize((out_size, out_size), resample=Image.NEAREST)
    img = Image.merge('RGB', (img, img, img))
    return img


def convert_wm811k(npy_dir, meta_json, out_img_dir, out_jsonl, image_size=512):
    os.makedirs(out_img_dir, exist_ok=True)
    with open(meta_json, 'r') as f:
        meta = json.load(f)
    # meta 期望格式：[{"id": str, "label": str, "npy": "path/to/array.npy"}, ...]
    with open(out_jsonl, 'w') as out_f:
        for item in meta:
            npy_path = os.path.join(npy_dir, item['npy'])
            arr = np.load(npy_path)  # 2D wafer map
            img = render_wafer_map(arr, out_size=image_size)
            img_name = f"{item['id']}.png"
            img_path = os.path.join(out_img_dir, img_name)
            img.save(img_path)
            # 计算连通域作为缺陷 bbox
            comps = connected_components((arr > 0).astype(np.uint8))
            defects = []
            for (x1, y1, x2, y2) in comps:
                # 缩放到输出图像坐标
                sx = image_size / arr.shape[1]
                sy = image_size / arr.shape[0]
                xx1 = int(x1 * sx)
                yy1 = int(y1 * sy)
                xx2 = int(x2 * sx)
                yy2 = int(y2 * sy)
                d_type = WM811K_TO_DEFECT.get(item.get('label', 'Random'), 'PARTICLE')
                if d_type == 'NONE':
                    continue
                defects.append({"type": d_type, "bbox": [xx1, yy1, xx2, yy2]})
            rec = {"image_path": img_path, "defects": defects}
            out_f.write(json.dumps(rec) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npy_dir', required=True, help='WM811K 原始 npy 存放目录')
    ap.add_argument('--meta_json', required=True, help='包含 id/label/npy 的元数据 JSON')
    ap.add_argument('--out_img_dir', required=True, help='输出图像目录')
    ap.add_argument('--out_jsonl', required=True, help='输出 JSONL 路径')
    ap.add_argument('--image_size', type=int, default=512)
    args = ap.parse_args()
    convert_wm811k(args.npy_dir, args.meta_json, args.out_img_dir, args.out_jsonl, image_size=args.image_size)


if __name__ == '__main__':
    main()