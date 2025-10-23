import json
from typing import Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class OCRTextDataset(Dataset):
    """
    JSONL format per line:
    {"image_path": "/abs/path/to/img.png", "text": "..."}
    - 用于 DeepEncoder 预训练的图像-文本对（OCR1.0/OCR2.0/LAION抽样）
    """

    def __init__(self, jsonl_path: str, image_size: Tuple[int, int] = (512, 512)):
        super().__init__()
        self.items = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.items.append(json.loads(line))
        self.W, self.H = image_size
        self.tfm = T.Compose([T.Resize(image_size), T.ToTensor()])
        # 文本编码器（open_clip tokenize 作为优先；失败则用简单字符编码）
        try:
            import open_clip
            self.tokenize = open_clip.tokenize
            self.use_open_clip = True
        except Exception:
            self.use_open_clip = False

    def __len__(self):
        return len(self.items)

    def simple_tokenize(self, text: str):
        # 退化方案：将字符转为 ASCII 序列；仅用于占位
        ids = [ord(c) % 256 for c in text[:512]]
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        item = self.items[idx]
        img = Image.open(item["image_path"]).convert("RGB")
        img_t = self.tfm(img)
        txt = item.get("text", "")
        if self.use_open_clip:
            txt_ids = self.tokenize([txt])[0]
        else:
            txt_ids = self.simple_tokenize(txt)
        return {"image": img_t, "text_ids": txt_ids}