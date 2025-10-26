# WaferSeeker 操作指南（数据处理 · 训练 · 推理）

本指南面向 WM811K 与生产 lot（5×5）晶圆场景，完整说明数据准备、训练与推理流程。项目支持可插拔主干（`simple/CLIP/SAM`）、MoE 解码器与长序列（位置嵌入 4096）。

## 概览
- 任务：将晶圆图像结构化为缺陷序列（类型 + bbox）。
- 关键模块：
  - `tools/wm811k_to_jsonl.py`：WM811K 转换为图像 + JSONL。
  - `waferseeker/data/dataset.py`：读取单片或 lot JSONL（自动展平）。
  - 训练脚本：`train.py`（单任务），`pretrain_encoder.py`（两阶段第1步），`train_joint.py`（两阶段第2步）。
  - 推理脚本：`infer.py`（支持单片与 lot 目录批量）。

## 环境准备
- Python 3.9+，安装 `torch`、`torchvision`。
- 可选：`open_clip`（CLIP 视觉/文本塔）、`segment_anything`（SAM）。
- 进入项目目录：`/Users/tohnee/workspace/start-up/WaferSeeker`。

## 数据规范
- 单片 JSONL 行：
  - `{ "image_path": "/abs/path/wafer.png", "defects": [{"type": "SCRATCH", "bbox": [x1,y1,x2,y2]}] }`
- lot JSONL 行（自动展平为单片样本）：
  - `{ "lot_id": "LOT123", "images": ["/abs/path/w0.png", ..., "/abs/path/w24.png"], "defects": [[{...}], ..., [{...}]] }`
- 图文对 JSONL（用于 DeepEncoder 预训练/联合训练）：
  - `{ "image_path": "/abs/path/img.jpg", "text": "某段文字" }`

## WM811K 转换
- 说明：将 `WM811K` 的 `npy` 晶圆图渲染为 `PNG`，用连通域生成缺陷 bbox，并映射常见标签到类型。
- 运行命令：
  - `python tools/wm811k_to_jsonl.py --npy_dir /data/WM811K/npys --meta_json /data/WM811K/meta.json --out_img_dir /data/wafer_imgs --out_jsonl /data/wafer.jsonl --image_size 512`
- 输出：
  - `images/*.png`（渲染图）；`wafer.jsonl`（每行含 `image_path` 与 `defects`）。

## lot（5×5）数据处理
- 数据集实现：`waferseeker/data/dataset.py` 支持 lot JSONL 自动展平为 25 个单片样本（按 `images` 与可选 `defects` 对齐）。
- 训练脚本无需变更；传入 lot JSONL 即可按单片训练。
- 推理支持 `--lot_dir`：输入目录中最多取 25 张晶圆图批量推理。

## 训练流程

### 1）单任务（仅 WM811K）
- 脚本：`train.py`
- 示例命令：
- `python train.py --jsonl /data/wafer.jsonl --backbone clip --decoder moe --d_model 512 --dec_layers 6 --dim_ff 2048 --batch_size 32 --epochs 10 --lr 1e-4 --save waferseeker.pt`
- 说明：支持 `simple/clip/sam` 主干与 `transformer/moe` 解码器；含余弦退火与梯度累积；保存 checkpoint 时写入 `backbone/decoder` 配置。

### 2）两阶段（第1阶段：DeepEncoder 预训练）
- 脚本：`pretrain_encoder.py`
- 数据：OCR1.0、OCR2.0、LAION 抽样（图文对）。
- 规范：2 epoch；`BS≈1280`（用梯度累积模拟）；`AdamW`；cosine LR=`5e-5`；`max_len=4096`。
- 命令示例：
- `python pretrain_encoder.py --image_jsonl /data/ocr1.jsonl /data/ocr2.jsonl /data/laion100m.jsonl --text_jsonl /data/ocr1.jsonl /data/ocr2.jsonl /data/laion100m.jsonl --backbone clip --epochs 2 --batch_size 128 --accum 10 --lr 5e-5 --max_len 4096 --save deepencoder.pt`
- 说明：`accum=10 × batch_size=128 ≈ 1280`；可切换 `--backbone sam`；无 `open_clip` 时自动降级简单编码方式。

### 3）两阶段（第2阶段：全参数联合训练）
- 脚本：`train_joint.py`
- 组成与占比：OCR1.0≈45%、OCR2.0≈25%、通用视觉≈20%、纯文本≈10%。
- 命令示例：
- `python train_joint.py --wm_jsonl /data/wafer.jsonl --ocr1_jsonl /data/ocr1.jsonl --ocr2_jsonl /data/ocr2.jsonl --vision_jsonl /data/vision.jsonl --text_jsonl /data/text.jsonl --backbone clip --decoder moe --d_model 768 --nhead 12 --dec_layers 12 --dim_ff 3072 --epochs 5 --batch_size 64 --accum 8 --lr 2e-4 --save joint.pt`
- 说明：支持混合采样与余弦退火；默认将文本批次训练为“无缺陷占位”序列以注入语言能力，可根据需要扩展为真实 OCR 序列。

## 推理

### 单片推理
- 命令：
- `python infer.py --ckpt joint.pt --image /path/to/wafer.png --strategy beam --beam_size 5 --out pred.json`
- 输出：`{"image": "...", "defects": [{"type": "...", "bbox": [x1,y1,x2,y2]}] }`

### lot 目录推理（批量 25 张）
- 命令：
- `python infer.py --ckpt joint.pt --lot_dir /path/to/lot_dir --strategy beam --beam_size 5 --out lot_result.json`
- 输出：`{"lot_dir": "...", "results": [{"image": "...", "defects": [...]}, ...] }`
- 说明：从 checkpoint 自动读取 `backbone/decoder/d_model/层数/dim_ff` 等配置，训练-推理一致。

## 评估与度量
- 模块：`waferseeker/utils/metrics.py`
- 用法：
  - `bbox_iou(b1, b2)`：返回两 bbox 的 IoU。
  - `match_and_score(preds, gts, iou_thresh=0.5)`：按类型与 IoU 阈值匹配，输出 Precision/Recall/F1。
- 建议：在验证集上评估缺陷级 P/R/F1，报告误报/漏报类别分布和得分随 IoU 阈值变化。

## 超参数与建议
- 主干：生产中优先 `clip`；支持 `sam`；`simple` 为降级备选。
- 解码器：MoE 对大容量更友好；`dim_ff` 随 `d_model` 线性扩展（如 `768→3072`）。
- 长序列：当单片缺陷密度高时增大 `max_seq_len`；解码器位置嵌入已至 4096。
- 优化器与调度器：`AdamW` + cosine；显存受限时提高 `--accum` 替代增大 batch。
- 坐标 bins：默认 `1024`；确保与图像归一化一致（脚本已处理）。

## 常见问题
- 依赖不可用：缺 `open_clip` 或 `segment_anything` 时自动降级为 `SimpleCNN`；建议安装后重试。
- JSONL 读取失败：检查每行是合法 JSON；字段名严格为 `image_path`/`defects` 或 `images`/`defects`；路径为可解析绝对/相对路径。
- 推理慢：改用 `--strategy greedy` 或减小 `d_model/层数`；MoE 可调低专家数或门控温度。
- 显存不足：降低 `batch_size` 或提高 `--accum`；减少 MoE 的 `dim_ff` 或层数。

## 文件结构与入口
- 训练入口：`train.py`（WM811K）、`pretrain_encoder.py`（两阶段第1步）、`train_joint.py`（两阶段第2步）。
- 推理入口：`infer.py`（单片/lot）。
- 数据转换：`tools/wm811k_to_jsonl.py`。
- 数据集：`waferseeker/data/dataset.py`（支持 lot 展平）。

## 快速检查清单
- 数据：WM811K 已转换为 `PNG + JSONL`；lot JSONL/目录结构规范。
- 训练：主干与解码器配置正确；`d_model/nhead/层数/dim_ff` 与资源匹配；梯度累积设置合理。
- 推理：checkpoint 保存了 `backbone/decoder` 配置；lot 输出按 5×5 可视化与统计。
- 评估：IoU 阈值与类别映射合理；验证集指标达标。