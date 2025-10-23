import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int = 4, top_k: int = 1, dropout: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x):
        # x: (B, L, D)
        B, L, D = x.shape
        gate_logits = self.gate(x)  # (B, L, E)
        gate_scores = F.softmax(gate_logits, dim=-1)
        if self.top_k == 1:
            idx = gate_scores.argmax(dim=-1)  # (B, L)
            out = torch.zeros_like(x)
            for e in range(self.num_experts):
                mask = (idx == e).float().unsqueeze(-1)
                if mask.sum() == 0:
                    continue
                out += self.experts[e](x) * mask
            return out
        else:
            topk_scores, topk_idx = torch.topk(gate_scores, self.top_k, dim=-1)  # (B, L, K)
            out = torch.zeros_like(x)
            for k in range(self.top_k):
                e_idx = topk_idx[..., k]
                score = topk_scores[..., k].unsqueeze(-1)
                for e in range(self.num_experts):
                    mask = (e_idx == e).float().unsqueeze(-1)
                    if mask.sum() == 0:
                        continue
                    out += self.experts[e](x) * mask * score
            return out


class MoETransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.1, num_experts: int = 4, top_k: int = 1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.moe_ff = MoEFeedForward(d_model, dim_ff, num_experts=num_experts, top_k=top_k, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Self-attention with causal mask
        x2, _ = self.self_attn(x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        x = x + self.dropout(x2)
        x = self.norm1(x)
        # Cross-attention to memory
        x2, _ = self.cross_attn(x, memory, memory, key_padding_mask=memory_key_padding_mask)
        x = x + self.dropout(x2)
        x = self.norm2(x)
        # MoE FFN
        x2 = self.moe_ff(x)
        x = x + self.dropout(x2)
        x = self.norm3(x)
        return x


class SequenceDecoderMoE(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, nhead: int = 8, num_layers: int = 6, dim_ff: int = 2048, dropout: float = 0.1, num_experts: int = 8, top_k: int = 1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(4096, d_model)
        self.layers = nn.ModuleList([
            MoETransformerDecoderLayer(d_model, nhead, dim_ff, dropout=dropout, num_experts=num_experts, top_k=top_k)
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, tgt_tokens: torch.Tensor, memory: torch.Tensor, tgt_key_padding_mask: torch.Tensor = None, memory_key_padding_mask: torch.Tensor = None):
        B, L = tgt_tokens.shape
        pos = torch.arange(L, device=tgt_tokens.device).unsqueeze(0).expand(B, L)
        x = self.tok_embed(tgt_tokens) + self.pos_embed(pos)
        causal_mask = torch.full((L, L), float('-inf'), device=tgt_tokens.device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=causal_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        logits = self.out_proj(x)
        return logits

    @torch.no_grad()
    def generate(self, memory: torch.Tensor, sos_id: int, eos_id: int, pad_id: int, max_len: int = 256, greedy: bool = True, beam_size: int = 3):
        B = memory.size(0)
        device = memory.device
        seq = torch.full((B, 1), sos_id, device=device, dtype=torch.long)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        for _ in range(max_len - 1):
            logits = self.forward(seq, memory)
            next_logits = logits[:, -1, :]
            next_id = next_logits.argmax(dim=-1)
            seq = torch.cat([seq, next_id[:, None]], dim=1)
            finished = finished | (next_id == eos_id)
            if finished.all():
                break
        for i in range(B):
            if not finished[i]:
                seq[i, -1] = eos_id
        return seq