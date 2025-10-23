from typing import List, Dict
import numpy as np


def bbox_iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    a2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    union = a1 + a2 - inter + 1e-6
    return inter / union


def match_defects(pred: List[Dict], gt: List[Dict], iou_thr: float = 0.5):
    used = set()
    tp = 0
    for p in pred:
        best = -1
        best_iou = 0
        for j, g in enumerate(gt):
            if j in used:
                continue
            if p["type"] != g["type"]:
                continue
            iou = bbox_iou(p["bbox"], g["bbox"])
            if iou > best_iou:
                best_iou = iou
                best = j
        if best != -1 and best_iou >= iou_thr:
            tp += 1
            used.add(best)
    fp = max(0, len(pred) - tp)
    fn = max(0, len(gt) - len(used))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-6, prec + rec)
    return {"precision": prec, "recall": rec, "f1": f1}