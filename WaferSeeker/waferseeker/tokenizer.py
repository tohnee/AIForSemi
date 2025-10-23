from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


@dataclass
class TokenizerConfig:
    defect_types: List[str]
    coord_bins: int = 1024  # coordinate quantization bins (image normalized to this grid)
    max_seq_len: int = 512


class DefectTokenizer:
    """
    Sequence design:
    [SOS] ([DEFECT_TYPE] [X1] [Y1] [X2] [Y2] [SEP])* [EOS]
    If no defect: [SOS] [NONE] [EOS]
    - DEFECT_TYPE from a controlled vocabulary
    - Coordinates are quantized to 0..coord_bins-1 after resizing to a fixed image size
    """

    PAD = 0
    SOS = 1
    EOS = 2
    SEP = 3
    NONE = 4

    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.types = list(config.defect_types)
        # Build vocabulary
        self.vocab: Dict[str, int] = {
            "<PAD>": self.PAD,
            "<SOS>": self.SOS,
            "<EOS>": self.EOS,
            "<SEP>": self.SEP,
            "<NONE>": self.NONE,
        }
        # defect type tokens
        self.type_offset = len(self.vocab)
        for i, t in enumerate(self.types):
            self.vocab[f"TYPE_{t}"] = self.type_offset + i
        self.num_type_tokens = len(self.types)
        # coordinate tokens
        self.coord_offset = self.type_offset + self.num_type_tokens
        for i in range(self.config.coord_bins):
            self.vocab[f"COORD_{i}"] = self.coord_offset + i
        self.vocab_size = len(self.vocab)
        # inverse mapping
        self.id2tok = {v: k for k, v in self.vocab.items()}

    def coord_to_id(self, v: int) -> int:
        if v < 0:
            v = 0
        if v >= self.config.coord_bins:
            v = self.config.coord_bins - 1
        return self.coord_offset + v

    def type_to_id(self, t: str) -> int:
        key = f"TYPE_{t}"
        if key not in self.vocab:
            raise KeyError(f"Unknown defect type: {t}")
        return self.vocab[key]

    def encode_defects(
        self,
        defects: List[Dict],
        image_size: Tuple[int, int],
    ) -> List[int]:
        """
        defects: list of {"type": str, "bbox": [x1, y1, x2, y2]}
        image_size: (W, H), coordinates assumed in pixel space of the resized image
        """
        seq: List[int] = [self.SOS]
        if len(defects) == 0:
            seq += [self.NONE, self.EOS]
            return seq
        for d in defects:
            t = d["type"]
            x1, y1, x2, y2 = d["bbox"]
            # quantize coordinates to bins
            W, H = image_size
            qx1 = int(round(x1 / (W - 1) * (self.config.coord_bins - 1)))
            qy1 = int(round(y1 / (H - 1) * (self.config.coord_bins - 1)))
            qx2 = int(round(x2 / (W - 1) * (self.config.coord_bins - 1)))
            qy2 = int(round(y2 / (H - 1) * (self.config.coord_bins - 1)))
            seq.append(self.type_to_id(t))
            seq.append(self.coord_to_id(qx1))
            seq.append(self.coord_to_id(qy1))
            seq.append(self.coord_to_id(qx2))
            seq.append(self.coord_to_id(qy2))
            seq.append(self.SEP)
            if len(seq) >= self.config.max_seq_len - 1:
                break
        # replace trailing SEP with EOS
        if seq[-1] == self.SEP:
            seq[-1] = self.EOS
        else:
            seq.append(self.EOS)
        return seq

    def pad(self, seq: List[int], max_len: Optional[int] = None) -> Tuple[List[int], List[int]]:
        if max_len is None:
            max_len = self.config.max_seq_len
        if len(seq) > max_len:
            seq = seq[:max_len]
        pad_len = max_len - len(seq)
        attn_mask = [1] * len(seq) + [0] * pad_len
        seq = seq + [self.PAD] * pad_len
        return seq, attn_mask

    def decode(self, ids: List[int], image_size: Tuple[int, int]) -> List[Dict]:
        """Convert token ids back to defects list.
        This will stop at EOS and ignore PAD tokens.
        """
        W, H = image_size
        defects: List[Dict] = []
        i = 0
        # skip leading SOS if present
        if i < len(ids) and ids[i] == self.SOS:
            i += 1
        if i < len(ids) and ids[i] == self.NONE:
            return []
        while i < len(ids):
            if ids[i] == self.EOS:
                break
            tok = ids[i]
            if tok in (self.PAD, self.SEP):
                i += 1
                continue
            # type
            if tok < self.coord_offset:
                t = self.id2tok[tok].replace("TYPE_", "")
                # coords
                if i + 4 >= len(ids):
                    break
                qx1 = ids[i + 1] - self.coord_offset
                qy1 = ids[i + 2] - self.coord_offset
                qx2 = ids[i + 3] - self.coord_offset
                qy2 = ids[i + 4] - self.coord_offset
                # de-quantize
                x1 = qx1 / (self.config.coord_bins - 1) * (W - 1)
                y1 = qy1 / (self.config.coord_bins - 1) * (H - 1)
                x2 = qx2 / (self.config.coord_bins - 1) * (W - 1)
                y2 = qy2 / (self.config.coord_bins - 1) * (H - 1)
                defects.append({"type": t, "bbox": [x1, y1, x2, y2]})
                i += 5
            else:
                i += 1
        return defects

    def vocab_summary(self) -> Dict[str, int]:
        return {
            "vocab_size": self.vocab_size,
            "num_types": self.num_type_tokens,
            "num_coord_tokens": self.config.coord_bins,
        }