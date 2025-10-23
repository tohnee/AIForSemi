import torch
import torch.nn as nn
import torch.nn.functional as F


class WaferSeekerModel(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        pad_id: int = 0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_id = pad_id
        self.label_smoothing = label_smoothing

    def forward(
        self,
        images: torch.Tensor,
        tgt_tokens: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor = None,
    ):
        memory, mem_mask = self.encoder(images)
        logits = self.decoder(
            tgt_tokens,
            memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=None,
        )
        return logits

    def compute_loss(self, logits: torch.Tensor, tgt_tokens: torch.Tensor):
        # shift: predict next token
        pred = logits[:, :-1, :]  # (B, L-1, V)
        tgt = tgt_tokens[:, 1:]   # (B, L-1)
        B, Lm1, V = pred.shape
        pred = pred.reshape(B * Lm1, V)
        tgt = tgt.reshape(B * Lm1)
        if self.label_smoothing > 0:
            # label smoothing cross entropy
            n_classes = V
            log_probs = F.log_softmax(pred, dim=-1)
            with torch.no_grad():
                true_dist = torch.zeros_like(log_probs)
                true_dist.fill_(self.label_smoothing / (n_classes - 1))
                true_dist.scatter_(1, tgt.unsqueeze(1), 1.0 - self.label_smoothing)
                true_dist[tgt == self.pad_id] = 0
            loss = torch.mean(torch.sum(-true_dist * log_probs, dim=-1))
        else:
            loss = F.cross_entropy(pred, tgt, ignore_index=self.pad_id)
        return loss

    @torch.no_grad()
    def generate(self, images: torch.Tensor, sos_id: int, eos_id: int, pad_id: int, max_len: int = 128, greedy: bool = True, beam_size: int = 3):
        memory, _ = self.encoder(images)
        seq = self.decoder.generate(
            memory,
            sos_id=sos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            max_len=max_len,
            greedy=greedy,
            beam_size=beam_size,
        )
        return seq