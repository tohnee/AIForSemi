import argparse
import random
import torch
from torch.utils.data import DataLoader

from waferseeker.tokenizer import DefectTokenizer, TokenizerConfig
from waferseeker.models.encoder import VisionEncoder
from waferseeker.models.decoder import SequenceDecoder
from waferseeker.models.moe_decoder import SequenceDecoderMoE
from waferseeker.models.model import WaferSeekerModel
from waferseeker.data.dataset import WaferDefectDataset, collate_batch
from waferseeker.data.ocr_text_dataset import OCRTextDataset


def parse_args():
    p = argparse.ArgumentParser()
    # 数据
    p.add_argument('--wm_jsonl', type=str, required=True, help='WM811K 转换后的 JSONL')
    p.add_argument('--ocr1_jsonl', type=str, required=True)
    p.add_argument('--ocr2_jsonl', type=str, required=True)
    p.add_argument('--vision_jsonl', type=str, required=True)
    p.add_argument('--text_jsonl', type=str, required=True)
    # 模型
    p.add_argument('--backbone', type=str, default='clip', choices=['simple','clip','sam'])
    p.add_argument('--decoder', type=str, default='moe', choices=['transformer','moe'])
    p.add_argument('--image_size', type=int, default=512)
    p.add_argument('--num_tokens', type=int, default=100)
    p.add_argument('--d_model', type=int, default=512)
    p.add_argument('--nhead', type=int, default=8)
    p.add_argument('--dec_layers', type=int, default=6)
    p.add_argument('--dim_ff', type=int, default=2048)
    # 训练
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--bs', type=int, default=32)
    p.add_argument('--accum', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--save', type=str, default='waferseeker_joint.pt')
    # 混合占比（OCR1/OCR2/通用视觉/纯文本）
    p.add_argument('--mix', type=float, nargs=4, default=[0.45, 0.25, 0.20, 0.10])
    return p.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    defect_types = ['SCRATCH', 'PARTICLE', 'HOLE', 'DISCOLOR', 'BRIDGE', 'OPEN']
    tok = DefectTokenizer(TokenizerConfig(defect_types=defect_types, coord_bins=1024, max_seq_len=256))

    enc = VisionEncoder(d_model=args.d_model, num_tokens=args.num_tokens, backbone=args.backbone).to(device)
    if args.decoder == 'moe':
        dec = SequenceDecoderMoE(vocab_size=tok.vocab_size, d_model=args.d_model, nhead=args.nhead, num_layers=args.dec_layers, dim_ff=args.dim_ff)
    else:
        dec = SequenceDecoder(vocab_size=tok.vocab_size, d_model=args.d_model, nhead=args.nhead, num_layers=args.dec_layers, dim_ff=args.dim_ff)
    model = WaferSeekerModel(enc, dec, pad_id=tok.PAD).to(device)

    ds_wm = WaferDefectDataset(args.wm_jsonl, tok, image_size=(args.image_size, args.image_size), augment=True, max_seq_len=256)
    dl_wm = DataLoader(ds_wm, batch_size=args.bs, shuffle=True, num_workers=2, collate_fn=collate_batch)

    ds_ocr1 = OCRTextDataset(args.ocr1_jsonl, image_size=(args.image_size, args.image_size))
    ds_ocr2 = OCRTextDataset(args.ocr2_jsonl, image_size=(args.image_size, args.image_size))
    ds_vis = OCRTextDataset(args.vision_jsonl, image_size=(args.image_size, args.image_size))
    ds_txt = OCRTextDataset(args.text_jsonl, image_size=(args.image_size, args.image_size))
    dls_text = [
        DataLoader(ds_ocr1, batch_size=args.bs, shuffle=True, num_workers=2),
        DataLoader(ds_ocr2, batch_size=args.bs, shuffle=True, num_workers=2),
        DataLoader(ds_vis, batch_size=args.bs, shuffle=True, num_workers=2),
        DataLoader(ds_txt, batch_size=args.bs, shuffle=True, num_workers=2),
    ]
    iters_text = [iter(dl) for dl in dls_text]

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device=='cuda'))

    # 简易 cosine annealing（每步更新）
    total_steps = args.epochs * (len(dl_wm) + sum(len(dl) for dl in dls_text))
    step = 0

    def cosine_lr(step, total, lr_max):
        import math
        return 0.5 * (1 + math.cos(math.pi * step / total)) * lr_max

    model.train()
    for epoch in range(args.epochs):
        # 迭代 WM811K 与混合的 OCR/视觉/文本，按比例采样
        mix_weights = args.mix
        for batch_wm in dl_wm:
            images_wm, tgts_wm, masks_wm = batch_wm
            images_wm = images_wm.to(device)
            tgts_wm = tgts_wm.to(device)
            masks_wm = masks_wm.to(device)
            # 先做一次 WM811K 的优化步
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                logits = model(images_wm, tgts_wm, tgt_key_padding_mask=masks_wm)
                loss_wm = model.compute_loss(logits, tgts_wm)
            scaler.scale(loss_wm).backward()
            if (step + 1) % args.accum == 0:
                lr_cur = cosine_lr(step, total_steps, args.lr)
                for g in opt.param_groups:
                    g['lr'] = lr_cur
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            step += 1

            # 按比例取一个文本批次（图像-文本对），用于语言能力与通用视觉注入
            pick = random.choices(range(4), weights=mix_weights, k=1)[0]
            try:
                batch_txt = next(iters_text[pick])
            except StopIteration:
                iters_text[pick] = iter(dls_text[pick])
                batch_txt = next(iters_text[pick])
            images_t = batch_txt['image'].to(device)
            # 文本只用于目标序列的语言能力占位，这里简化为“无缺陷”序列训练（语言侧），也可以扩展为真实 OCR 序列
            none_seq = [tok.SOS, tok.NONE, tok.EOS]
            none_seq, none_mask = tok.pad(none_seq, max_len=256)
            tgts_t = torch.tensor([none_seq] * images_t.size(0), dtype=torch.long, device=device)
            masks_t = torch.tensor([none_mask] * images_t.size(0), dtype=torch.bool, device=device)

            with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                logits_t = model(images_t, tgts_t, tgt_key_padding_mask=masks_t)
                loss_t = model.compute_loss(logits_t, tgts_t)
            scaler.scale(loss_t).backward()
            if (step + 1) % args.accum == 0:
                lr_cur = cosine_lr(step, total_steps, args.lr)
                for g in opt.param_groups:
                    g['lr'] = lr_cur
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            step += 1

        print(f"Epoch {epoch+1}: loss_wm={loss_wm.item():.4f}, loss_t={loss_t.item():.4f}")

    torch.save({
        'model': model.state_dict(),
        'config': vars(args),
        'vocab': tok.vocab,
        'backbone': args.backbone,
        'decoder': args.decoder,
    }, args.save)


if __name__ == '__main__':
    main()