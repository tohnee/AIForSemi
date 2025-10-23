import argparse
import math
import torch
from torch.utils.data import DataLoader

from waferseeker.tokenizer import DefectTokenizer, TokenizerConfig
from waferseeker.models.encoder import VisionEncoder
from waferseeker.models.decoder import SequenceDecoder
from waferseeker.models.moe_decoder import SequenceDecoderMoE
from waferseeker.models.model import WaferSeekerModel
from waferseeker.data.dataset import WaferDefectDataset, collate_batch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--jsonl', type=str, required=True, help='Path to training JSONL')
    p.add_argument('--image_size', type=int, default=512)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--coord_bins', type=int, default=1024)
    p.add_argument('--max_seq_len', type=int, default=256)
    p.add_argument('--num_tokens', type=int, default=100)
    p.add_argument('--d_model', type=int, default=512)
    p.add_argument('--nhead', type=int, default=8)
    p.add_argument('--dec_layers', type=int, default=6)
    p.add_argument('--dim_ff', type=int, default=2048)
    p.add_argument('--decoder', type=str, default='moe', choices=['transformer','moe'])
    p.add_argument('--backbone', type=str, default='clip', choices=['simple','clip','sam'])
    p.add_argument('--accum', type=int, default=4, help='gradient accumulation steps')
    p.add_argument('--save', type=str, default='waferseeker.pt')
    return p.parse_args()


def cosine_lr(step, total, lr_max):
    return 0.5 * (1 + math.cos(math.pi * step / total)) * lr_max


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    defect_types = ['SCRATCH', 'PARTICLE', 'HOLE', 'DISCOLOR', 'BRIDGE', 'OPEN']
    tok = DefectTokenizer(TokenizerConfig(defect_types=defect_types, coord_bins=args.coord_bins, max_seq_len=args.max_seq_len))

    enc = VisionEncoder(d_model=args.d_model, num_tokens=args.num_tokens, backbone=args.backbone)
    if args.decoder == 'moe':
        dec = SequenceDecoderMoE(vocab_size=tok.vocab_size, d_model=args.d_model, nhead=args.nhead, num_layers=args.dec_layers, dim_ff=args.dim_ff)
    else:
        dec = SequenceDecoder(vocab_size=tok.vocab_size, d_model=args.d_model, nhead=args.nhead, num_layers=args.dec_layers, dim_ff=args.dim_ff)
    model = WaferSeekerModel(enc, dec, pad_id=tok.PAD).to(device)

    ds = WaferDefectDataset(args.jsonl, tok, image_size=(args.image_size, args.image_size), augment=True, max_seq_len=args.max_seq_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_batch)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device=='cuda'))

    total_steps = len(dl) * args.epochs
    step = 0

    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for images, tgts, masks in dl:
            images = images.to(device)
            tgts = tgts.to(device)
            masks = masks.to(device)
            with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                logits = model(images, tgts, tgt_key_padding_mask=masks)
                loss = model.compute_loss(logits, tgts)
            scaler.scale(loss).backward()
            if (step + 1) % args.accum == 0:
                lr_cur = cosine_lr(step, total_steps, args.lr)
                for g in opt.param_groups:
                    g['lr'] = lr_cur
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            step += 1
            total += loss.item()
        print(f"Epoch {epoch+1}: loss={total/len(dl):.4f}")
    torch.save({'model': model.state_dict(), 'config': vars(args), 'vocab': tok.vocab, 'backbone': args.backbone, 'decoder': args.decoder, 'dim_ff': args.dim_ff}, args.save)


if __name__ == '__main__':
    main()