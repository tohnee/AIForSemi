import math
import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int = 3, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, hidden, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CLIPBackbone(nn.Module):
    def __init__(self, out_channels: int = 768):
        super().__init__()
        self.available = False
        try:
            import open_clip
            self.available = True
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
            self.model = model.visual
            self.out_channels = self.model.output_dim
        except Exception:
            self.model = None
            self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.available:
            return self.model(x)
        # fallback: simple conv to match channel size
        return SimpleCNN(in_channels=x.size(1), hidden=self.out_channels)(x)


class SAMBackbone(nn.Module):
    def __init__(self, out_channels: int = 256):
        super().__init__()
        self.available = False
        try:
            from segment_anything import sam_model_registry
            self.available = True
            sam = sam_model_registry.get('vit_b')(checkpoint=None)
            self.model = sam.image_encoder
            self.out_channels = 256
        except Exception:
            self.model = None
            self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.available:
            return self.model(x)
        return SimpleCNN(in_channels=x.size(1), hidden=self.out_channels)(x)


class VisionEncoder(nn.Module):
    """
    Image encoder producing a fixed number of vision tokens.
    - backbone: configurable (simple CNN / CLIP / SAM)
    - compress: adaptive pooling to grid (Gh, Gw) s.t. Gh*Gw = num_tokens, then project to d_model
    """

    def __init__(
        self,
        d_model: int = 256,
        num_tokens: int = 100,
        in_channels: int = 3,
        backbone: str = 'simple',
    ):
        super().__init__()
        if backbone == 'clip':
            self.backbone = CLIPBackbone()
            in_ch = getattr(self.backbone, 'out_channels', 768)
        elif backbone == 'sam':
            self.backbone = SAMBackbone()
            in_ch = getattr(self.backbone, 'out_channels', 256)
        else:
            self.backbone = SimpleCNN(in_channels=in_channels, hidden=256)
            in_ch = 256
        self.proj = nn.Conv2d(in_ch, d_model, kernel_size=1)
        gh = int(math.sqrt(num_tokens))
        gw = max(1, num_tokens // gh)
        self.gh, self.gw = gh, gw
        self.pool = nn.AdaptiveAvgPool2d((gh, gw))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        if feat.dim() == 3:  # some backbones may return (B, T, C)
            B, T, C = feat.shape
            Gh = int(math.sqrt(T))
            Gw = max(1, T // Gh)
            feat = feat.transpose(1, 2).view(B, C, Gh, Gw)
        feat = self.proj(feat)
        feat = self.pool(feat)
        B, D, Gh, Gw = feat.shape
        tokens = feat.view(B, D, Gh * Gw).permute(0, 2, 1)
        tokens = self.norm(tokens)
        mem_mask = torch.ones(B, Gh * Gw, dtype=torch.bool, device=tokens.device)
        return tokens, mem_mask