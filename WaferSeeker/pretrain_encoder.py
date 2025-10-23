import argparse
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from waferseeker.models.encoder import VisionEncoder
from waferseeker.data.ocr_text_dataset import OCRTextDataset


class TextTower(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        self.available = False
        try:
            import open_clip
            self.available = True
            self.model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
            self.text = self.model.text
            self.out_dim = self.model.text_projection.shape[-1]
        except Exception:
            self.text = None
            self.out_dim = out_dim
            self.emb = nn.Embedding(512, out_dim)
            self.rnn = nn.GRU(out_dim, out_dim, batch_first=True)

    def forward(self, text_ids):
        if self.available:
            x = self.text(text_ids)
            return x
        else:
            # 简化：字符 id -> embedding -> GRU -> 均值池化
            x = self.emb(text_ids)
            x, _ = self.rnn(x)
            x = x.mean(dim=1)
            return x


class ImageTower(nn.Module):
    def __init__(self, encoder: VisionEncoder, out_dim=512):
        super().__init__()
        self.encoder = encoder
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(encoder.norm.normalized_shape[0], out_dim)

    def forward(self, images):
        mem, _ = self.encoder(images)
        # mem: (B, T, D) -> pool over T
        x = mem.transpose(1, 2)  # (B, D, T)
        x = self.pool(x).squeeze(-1)  # (B, D)
        x = self.proj(x)
        return x


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ocr1_jsonl', required=True)
    p.add_argument('--ocr2_jsonl', required=True)
    p.add_argument('--laion_jsonl', required=True)
    p.add_argument('--image_size', type=int, default=512)
    p.add_argument('--backbone', type=str, default='clip', choices=['simple','clip','sam'])
    p.add_argument('--num_tokens', type=int, default=100)
    p.add_argument('--d_model', type=int, default=512)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--bs', type=int, default=128)
    p.add_argument('--accum', type=int, default=10, help='梯度累积步数以模拟大批次，例如128*10≈1280')
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--save', type=str, default='deepencoder.pt')
    return p.parse_args()


def cosine_annealing(step, total_steps, lr_max):
    return 0.5 * (1 + math.cos(math.pi * step / total_steps)) * lr_max


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ds_ocr1 = OCRTextDataset(args.ocr1_jsonl, image_size=(args.image_size, args.image_size))
    ds_ocr2 = OCRTextDataset(args.ocr2_jsonl, image_size=(args.image_size, args.image_size))
    ds_laion = OCRTextDataset(args.laion_jsonl, image_size=(args.image_size, args.image_size))
    ds = ConcatDataset([ds_ocr1, ds_ocr2, ds_laion])
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=4)

    enc = VisionEncoder(d_model=args.d_model, num_tokens=args.num_tokens, backbone=args.backbone).to(device)
    img_tower = ImageTower(enc, out_dim=args.d_model).to(device)
    txt_tower = TextTower(out_dim=args.d_model).to(device)

    opt = torch.optim.AdamW(list(img_tower.parameters()) + list(enc.parameters()) + list(txt_tower.parameters()), lr=args.lr)

    total_steps = len(dl) * args.epochs
    step = 0
    scaler = torch.cuda.amp.GradScaler(enabled=(device=='cuda'))

    for epoch in range(args.epochs):
        for batch in dl:
            images = batch['image'].to(device)
            # open_clip 的 tokenize 已在数据集内部；如不可用则 simple tokenizer 返回不定长，需要 pad
            text_ids = batch['text_ids']
            if text_ids.dim() == 1:
                text_ids = torch.nn.utils.rnn.pad_sequence([text_ids], batch_first=True)
            text_ids = text_ids.to(device)

            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                img_feat = img_tower(images)   # (B, D)
                txt_feat = txt_tower(text_ids) # (B, D)
                img_feat = nn.functional.normalize(img_feat, dim=-1)
                txt_feat = nn.functional.normalize(txt_feat, dim=-1)
                logits = img_feat @ txt_feat.t()  # (B, B)
                targets = torch.arange(logits.size(0), device=device)
                loss_i = nn.functional.cross_entropy(logits, targets)
                loss_t = nn.functional.cross_entropy(logits.t(), targets)
                loss = (loss_i + loss_t) / 2
            scaler.scale(loss).backward()
            if (step + 1) % args.accum == 0:
                # Cosine Annealing 手动调整学习率
                lr_now = cosine_annealing(step, total_steps, args.lr)
                for g in opt.param_groups:
                    g['lr'] = lr_now
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            step += 1
        print(f"Epoch {epoch+1}: loss={loss.item():.4f}")

    torch.save({'encoder': enc.state_dict(), 'config': vars(args)}, args.save)


if __name__ == '__main__':
    main()