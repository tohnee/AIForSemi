import json
from typing import List, Dict, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from waferseeker.tokenizer import DefectTokenizer, TokenizerConfig


class WaferDefectDataset(Dataset):
    """
    JSONL format per line:
    {
      "image_path": "/abs/path/to/wafer.png",
      "defects": [{"type": "SCRATCH", "bbox": [x1,y1,x2,y2]}, ...]
    }
    or lot format:
    {
      "lot_id": "LOT123",
      "images": ["/abs/path/w0.png", ..., "/abs/path/w24.png"],
      "defects": [[{...}], ..., [{...}]]  # optional, aligned per wafer
    }
    The lot format will be flattened into per-wafer samples.
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: DefectTokenizer,
        image_size: Tuple[int, int] = (512, 512),
        augment: bool = True,
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.items = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                rec = json.loads(line)
                if isinstance(rec, dict) and 'images' in rec:
                    imgs = rec.get('images', [])
                    defects_list = rec.get('defects', [[] for _ in imgs])
                    for i, img_path in enumerate(imgs):
                        d = defects_list[i] if i < len(defects_list) else []
                        self.items.append({"image_path": img_path, "defects": d})
                else:
                    self.items.append(rec)
        self.W, self.H = image_size
        self.max_seq_len = max_seq_len
        tfms = [T.Resize(image_size), T.ToTensor()]
        if augment:
            tfms = [
                T.Resize(image_size),
                T.ColorJitter(brightness=0.05, contrast=0.05),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.ToTensor(),
            ]
        self.tfms = T.Compose(tfms)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img = Image.open(item["image_path"]).convert("RGB")
        img_t = self.tfms(img)
        defects = item.get("defects", [])
        seq = self.tokenizer.encode_defects(defects, (self.W, self.H))
        seq, attn = self.tokenizer.pad(seq, self.max_seq_len)
        return {
            "image": img_t,
            "tgt": torch.tensor(seq, dtype=torch.long),
            "tgt_mask": torch.tensor([a == 0 for a in attn], dtype=torch.bool),
        }


def collate_batch(batch: List[Dict]):
    images = torch.stack([b["image"] for b in batch], dim=0)
    tgts = torch.stack([b["tgt"] for b in batch], dim=0)
    masks = torch.stack([b["tgt_mask"] for b in batch], dim=0)
    return images, tgts, masks