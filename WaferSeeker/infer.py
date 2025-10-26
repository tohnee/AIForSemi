import argparse
import json
import os
import glob
import torch
from PIL import Image
import torchvision.transforms as T

from waferseeker.tokenizer import DefectTokenizer, TokenizerConfig
from waferseeker.models.encoder import VisionEncoder
from waferseeker.models.decoder import SequenceDecoder
from waferseeker.models.moe_decoder import SequenceDecoderMoE
from waferseeker.models.model import WaferSeekerModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--image', type=str, default=None, help='单张晶圆图路径')
    p.add_argument('--lot_dir', type=str, default=None, help='lot 目录（包含 25 张晶圆图）')
    p.add_argument('--image_size', type=int, default=512)
    p.add_argument('--max_len', type=int, default=256)
    p.add_argument('--strategy', type=str, default='greedy', choices=['greedy','beam'])
    p.add_argument('--beam_size', type=int, default=5)
    p.add_argument('--out', type=str, default='pred.json')
    return p.parse_args()


def load_model(args, device):
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get('config', {})
    defect_types = ['SCRATCH', 'PARTICLE', 'HOLE', 'DISCOLOR', 'BRIDGE', 'OPEN']
    tok = DefectTokenizer(TokenizerConfig(defect_types=defect_types, coord_bins=cfg.get('coord_bins', 1024), max_seq_len=cfg.get('max_seq_len', args.max_len)))
    enc = VisionEncoder(d_model=cfg.get('d_model', 512), num_tokens=cfg.get('num_tokens', 100), backbone=ckpt.get('backbone', 'simple'))
    if ckpt.get('decoder', 'transformer') == 'moe':
        dec = SequenceDecoderMoE(vocab_size=tok.vocab_size, d_model=cfg.get('d_model', 512), nhead=cfg.get('nhead', 8), num_layers=cfg.get('dec_layers', 6), dim_ff=cfg.get('dim_ff', 2048))
    else:
        dec = SequenceDecoder(vocab_size=tok.vocab_size, d_model=cfg.get('d_model', 512), nhead=cfg.get('nhead', 8), num_layers=cfg.get('dec_layers', 6), dim_ff=cfg.get('dim_ff', 2048))
    model = WaferSeekerModel(enc, dec, pad_id=tok.PAD)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()
    return model, tok


def infer_single(model, tok, image_path, image_size, max_len, strategy, beam_size, device):
    img = Image.open(image_path).convert('RGB')
    tfm = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])
    img_t = tfm(img).unsqueeze(0).to(device)
    greedy = (strategy == 'greedy')
    seq = model.generate(img_t, sos_id=tok.SOS, eos_id=tok.EOS, pad_id=tok.PAD, max_len=max_len, greedy=greedy, beam_size=beam_size)
    ids = seq.squeeze(0).tolist()
    defects = tok.decode(ids, (image_size, image_size))
    return {"image": image_path, "defects": defects}


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(args, device)

    if args.lot_dir:
        # 读取 lot 目录下的晶圆图片（按常见扩展名），默认取前 25 张并排序
        exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp')
        paths = []
        for e in exts:
            paths.extend(glob.glob(os.path.join(args.lot_dir, e)))
        paths = sorted(paths)[:25]
        results = []
        for p in paths:
            results.append(infer_single(model, tok, p, args.image_size, args.max_len, args.strategy, args.beam_size, device))
        out_obj = {"lot_dir": args.lot_dir, "results": results}
        with open(args.out, 'w') as f:
            json.dump(out_obj, f, indent=2)
        print(json.dumps(out_obj, indent=2))
        return

    if args.image:
        res = infer_single(model, tok, args.image, args.image_size, args.max_len, args.strategy, args.beam_size, device)
        with open(args.out, 'w') as f:
            json.dump(res, f, indent=2)
        print(json.dumps(res, indent=2))
        return

    raise ValueError('必须提供 --image 或 --lot_dir')


if __name__ == '__main__':
    main()