import argparse
import json
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
    p.add_argument('--image', type=str, required=True)
    p.add_argument('--image_size', type=int, default=512)
    p.add_argument('--max_len', type=int, default=256)
    p.add_argument('--beam', type=int, default=0, help='0=greedy, >0=beam size')
    p.add_argument('--output_json', type=str, default='pred.json')
    return p.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get('config', {})

    defect_types = ['SCRATCH', 'PARTICLE', 'HOLE', 'DISCOLOR', 'BRIDGE', 'OPEN']
    tok = DefectTokenizer(TokenizerConfig(defect_types=defect_types, coord_bins=cfg.get('coord_bins', 1024), max_seq_len=cfg.get('max_seq_len', 256)))

    enc = VisionEncoder(d_model=cfg.get('d_model', 256), num_tokens=cfg.get('num_tokens', 100), backbone=ckpt.get('backbone', 'simple'))
if ckpt.get('decoder', 'transformer') == 'moe':
    dec = SequenceDecoderMoE(vocab_size=tok.vocab_size, d_model=cfg.get('d_model', 256), nhead=cfg.get('nhead', 8), num_layers=cfg.get('dec_layers', 4), dim_ff=cfg.get('dim_ff', 2048))
else:
    dec = SequenceDecoder(vocab_size=tok.vocab_size, d_model=cfg.get('d_model', 256), nhead=cfg.get('nhead', 8), num_layers=cfg.get('dec_layers', 4))
model = WaferSeekerModel(enc, dec, pad_id=tok.PAD)
model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()

    img = Image.open(args.image).convert('RGB')
    tfm = T.Compose([T.Resize((args.image_size, args.image_size)), T.ToTensor()])
    img_t = tfm(img).unsqueeze(0).to(device)

    seq = model.generate(img_t, sos_id=tok.SOS, eos_id=tok.EOS, pad_id=tok.PAD, max_len=args.max_len, greedy=(args.beam==0), beam_size=max(1, args.beam))
    ids = seq.squeeze(0).tolist()
    defects = tok.decode(ids, (args.image_size, args.image_size))

    with open(args.output_json, 'w') as f:
        json.dump({"image": args.image, "defects": defects}, f, indent=2)
    print(json.dumps({"image": args.image, "defects": defects}, indent=2))


if __name__ == '__main__':
    main()