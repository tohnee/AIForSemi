import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceDecoder(nn.Module):
    """
    Transformer decoder with cross-attention over vision memory tokens.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(4096, d_model)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        tgt_tokens: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None,
    ):
        # tgt_tokens: (B, L)
        B, L = tgt_tokens.shape
        pos = torch.arange(L, device=tgt_tokens.device).unsqueeze(0).expand(B, L)
        x = self.tok_embed(tgt_tokens) + self.pos_embed(pos)
        # causal mask for autoregression
        causal_mask = torch.full((L, L), float('-inf'), device=tgt_tokens.device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        y = self.decoder(
            x,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        logits = self.out_proj(y)
        return logits

    @torch.no_grad()
    def generate(
        self,
        memory: torch.Tensor,
        sos_id: int,
        eos_id: int,
        pad_id: int,
        max_len: int = 128,
        greedy: bool = True,
        beam_size: int = 3,
    ):
        B = memory.size(0)
        device = memory.device
        if greedy:
            seq = torch.full((B, 1), sos_id, device=device, dtype=torch.long)
            finished = torch.zeros(B, dtype=torch.bool, device=device)
            for _ in range(max_len - 1):
                logits = self.forward(seq, memory)
                next_logits = logits[:, -1, :]  # (B, V)
                next_id = next_logits.argmax(dim=-1)
                seq = torch.cat([seq, next_id[:, None]], dim=1)
                finished = finished | (next_id == eos_id)
                if finished.all():
                    break
            # pad unfinished with eos
            for i in range(B):
                if not finished[i]:
                    seq[i, -1] = eos_id
            return seq
        else:
            # simple beam search (batch size 1 for simplicity)
            assert B == 1, "Beam search implemented for B=1 in this skeleton"
            beams = [(torch.tensor([sos_id], device=device).long(), 0.0)]
            for _ in range(max_len - 1):
                new_beams = []
                for seq, score in beams:
                    logits = self.forward(seq[None, :], memory)
                    next_logits = logits[:, -1, :].squeeze(0)
                    log_probs = F.log_softmax(next_logits, dim=-1)
                    topk = torch.topk(log_probs, beam_size)
                    for idx, lp in zip(topk.indices, topk.values):
                        new_seq = torch.cat([seq, idx.view(1)], dim=0)
                        new_score = score + lp.item()
                        new_beams.append((new_seq, new_score))
                # prune beams
                new_beams.sort(key=lambda x: x[1], reverse=True)
                beams = new_beams[:beam_size]
                # early stop if all beams ended
                if all(seq[-1].item() == eos_id for seq, _ in beams):
                    break
            best_seq = max(beams, key=lambda x: x[1])[0]
            return best_seq.view(1, -1)