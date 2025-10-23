# WaferSeeker

基于 Encoder-Decoder（可插拔视觉主干 + Transformer/MoE 解码器）的晶圆缺陷结构化识别系统。支持：
- 可插拔视觉主干：`simple`/`CLIP`/`SAM`，生产可替换更强主干；
- 大规模解码器：标准 Transformer 或 MoE（Top-k gating），可提升至接近 DeepSeek-OCR 规模；
- WM811K 数据集转换为图像 + 缺陷 JSONL；
- 两阶段训练流程：DeepEncoder 预训练（图文对）→ 全参数联合训练（缺陷序列生成）；
- 推理支持贪婪/Beam 并自动加载主干/解码器配置。

## 架构总览
- VisionEncoder（pluggable）
  - `SimpleCNN`：轻量卷积 + 线性投影。
  - `CLIPBackbone`：使用 `open_clip`，提取视觉特征；不可用时自动降级。
  - `SAMBackbone`：使用 `segment_anything`，提取图像 token；不可用时自动降级。
- SequenceDecoder / SequenceDecoderMoE
  - 位置嵌入长度调整为 `4096`，支持长序列训练；
  - MoE：`MoEFeedForward` + `Top-k` gating，支持大参数规模。
- WaferSeekerModel
  - Encoder 输出图像 token；Decoder 生成序列化缺陷结构；
  - `compute_loss` 为自回归交叉熵；`generate` 支持贪婪/Beam。

## Token 设计与序列格式
- 特殊：`[BOS] [EOS] [PAD]`
- 类型：`<TYPE:SCRATCH>` 等（可扩展）
- 坐标：`<X:i> <Y:j> <W:w> <H:h>`，均为离散化 bin（默认 1024）
- 示例序列：
  `[BOS] <TYPE:SCRATCH> <X:123> <Y:45> <W:20> <H:5> <TYPE:OPEN> <X:80> ... [EOS]`

## 数据准备
### WM811K → JSONL
使用转换脚本将 `.npy` 晶圆图转为 `PNG` + 缺陷 bbox。

```
python tools/wm811k_to_jsonl.py \
  --wm_root /path/to/WM811K/ \
  --out_dir /path/to/wafer_jsonl/ \
  --image_size 512 \
  --min_component 8
```
输出：
- `images/*.png`
- `wafer.jsonl`：每行含 `{image_path, defects: [{type, bbox}]}`。

### 图文对（OCR/通用/文本）
将 OCR1.0 / OCR2.0 / 通用图像-文本 与 纯文本 数据整理为 JSONL：

- 图像-文本 JSONL 行：`{"image": "/path/img.jpg", "text": "某段文字"}`
- 纯文本 JSONL 行：`{"text": "某段文字"}`（训练时将使用占位图或文本塔）

## 训练流程
### 阶段一：DeepEncoder 预训练（对齐用户规范）
- 数据：OCR1.0（43M）+ OCR2.0（16M）+ LAION 采样 100M（通用）
- 训练：2 epoch；`BS=1280`（使用梯度累积模拟）；`AdamW`；余弦退火；初始 `LR=5e-5`；`max_len=4096`

命令示例：
```
python pretrain_encoder.py \
  --image_jsonl /data/ocr1.jsonl /data/ocr2.jsonl /data/laion100m.jsonl \
  --text_jsonl  /data/ocr1.jsonl /data/ocr2.jsonl /data/laion100m.jsonl \
  --backbone clip \
  --epochs 2 \
  --batch_size 128 \
  --accum 10 \
  --lr 5e-5 \
  --max_len 4096 \
  --save deepencoder.pt
```
说明：
- `accum=10` × `batch_size=128` ≈ 1280；
- 可切换 `--backbone sam` 在有 `segment_anything` 环境下使用 SAM；
- 如无 `open_clip`，脚本自动降级为简单编码方式。

### 阶段二：全参数联合训练（混合采样）
- 组成与占比：
  - OCR1.0（约 45%）
  - OCR2.0（约 25%）
  - 通用视觉（20%）
  - 纯文本（10%）
- 模型：可选 `--decoder moe`（默认），支持大参数；可选主干 `--backbone clip/sam`；
- 优化器与调度器：`AdamW` + 余弦退火；支持梯度累积。

命令示例：
```
python train_joint.py \
  --wm_jsonl /data/wafer.jsonl \
  --ocr1_jsonl /data/ocr1.jsonl \
  --ocr2_jsonl /data/ocr2.jsonl \
  --vision_jsonl /data/vision.jsonl \
  --text_jsonl /data/text.jsonl \
  --backbone clip \
  --decoder moe \
  --d_model 768 \
  --nhead 12 \
  --dec_layers 12 \
  --dim_ff 3072 \
  --epochs 5 \
  --batch_size 64 \
  --accum 8 \
  --lr 2e-4 \
  --save joint.pt
```

### 单任务缺陷训练（仅 WM811K）
```
python train.py \
  --jsonl /data/wafer.jsonl \
  --backbone clip \
  --decoder moe \
  --d_model 512 \
  --dec_layers 6 \
  --dim_ff 2048 \
  --batch_size 32 \
  --epochs 10 \
  --lr 1e-4 \
  --save waferseeker.pt
```

## 推理
- 自动读取 checkpoint 中的 `backbone` 与 `decoder` 类型；
- 支持贪婪与 Beam 搜索；

```
python infer.py \
  --ckpt joint.pt \
  --image /path/to/wafer.png \
  --strategy beam \
  --beam_size 5 \
  --out result.json
```

## 评价与度量
- `waferseeker/utils/metrics.py` 提供 bbox IoU + 精/召/F1 缺陷匹配评估；

## 与 DeepSeek-OCR 的关系与差异
- 相同：Encoder-Decoder 思路，长序列支持，图文对预训练；
- 差异：任务为晶圆缺陷结构化，序列格式为类型 + 坐标；
- 扩展：MoE 解码、SAM/CLIP 主干、自定义采样策略、两阶段训练流程。

## 工程建议
- 大批次训练优先使用梯度累积；
- MoE 规模提升时需关注显存与路由稳定性；
- 数据管线统一转为 JSONL，便于混合采样与追踪；
- 尝试更精细的缺陷几何（多点/多边形）、图像金字塔、条件提示等扩展。