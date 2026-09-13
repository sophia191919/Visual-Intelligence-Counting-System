# -*- coding: utf-8 -*-
"""模型评估脚本：在验证集上计算 mAP + 生成混淆矩阵，量化「达标没达标」。

用法：
    python val.py              # 优先评估 best.pt；无则回退 yolov8n.pt（仅作基线参考）
    python val.py <权重路径>    # 评估指定权重

评估质量目标（见《准确率优化方案.md》）：
    mAP50(验证集) ≥ 0.85，Recall ≥ 0.90，Precision ≥ 0.85，
    摩托车 / 自行车 / 卡车 单类 F1 ≥ 0.75。

输出（自动写入 runs/detect/val/）：
    results.csv                       逐类 P/R/mAP 数据
    confusion_matrix.png              混淆矩阵
    confusion_matrix_normalized.png   归一化混淆矩阵
    PR_curve.png / F1_curve.png       精度召回曲线 / F1 曲线
"""
import os
import sys
import numpy as np
from ultralytics import YOLO

DATA_YAML = r"D:/视觉计数系统/Vehicle-counting.v11-roboflow-instant-2--eval-.yolov8/data.yaml"
BEST_PT = r"D:/视觉计数系统/runs/detect/train/weights/best.pt"

VAL_IMGSZ = 960   # 与训练 imgsz 保持一致
VAL_BATCH = 16
VAL_WORKERS = 0   # Windows 兼容

# 达标目标（与《准确率优化方案.md》一致）
TARGET_MAP50 = 0.85
TARGET_RECALL = 0.90
TARGET_PRECISION = 0.85


def _num(x):
    """把可能的 tensor / 数组转成标量（多类时取均值）。"""
    if x is None:
        return float("nan")
    if hasattr(x, "mean"):
        return float(np.mean(x))
    return float(x)


def main():
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    elif os.path.exists(BEST_PT):
        model_path = BEST_PT
    else:
        print("[提示] 未找到 best.pt，回退用 yolov8n.pt 评估（只能作基线参考，不代表训练后指标）。")
        model_path = "yolov8n.pt"

    print(f"[评估模块] 使用权重: {model_path}")
    model = YOLO(model_path)

    # 在验证集上评估，plots=True 生成混淆矩阵 / PR 曲线 / F1 曲线
    metrics = model.val(
        data=DATA_YAML,
        imgsz=VAL_IMGSZ,
        batch=VAL_BATCH,
        workers=VAL_WORKERS,
        plots=True,
        project="runs/detect",
        name="val",
    )

    # 汇总打印核心指标（兼容不同版本字段名）
    box = getattr(metrics, "box", None)
    if box is not None:
        map50 = _num(getattr(box, "map50", getattr(box, "ap50", None)))
        map5095 = _num(getattr(box, "map", getattr(box, "ap", None)))
        precision = _num(getattr(box, "mp", getattr(box, "p", None)))
        recall = _num(getattr(box, "mr", getattr(box, "r", None)))

        print("\n========== 评估汇总 ==========")
        print(f"mAP50     : {map50:.4f}")
        print(f"mAP50-95  : {map5095:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print("===============================")

        print("\n[达标判定]")
        print(f"  mAP50    ≥ {TARGET_MAP50:.2f} -> {'√ 达标' if map50 >= TARGET_MAP50 else '× 未达标'}")
        print(f"  Recall   ≥ {TARGET_RECALL:.2f} -> {'√ 达标' if recall >= TARGET_RECALL else '× 未达标'}")
        print(f"  Precision≥ {TARGET_PRECISION:.2f} -> {'√ 达标' if precision >= TARGET_PRECISION else '× 未达标'}")

    save_dir = os.path.join("runs", "detect", "val")
    print(f"\n[完成] 混淆矩阵等图表已保存至: {save_dir}")


if __name__ == "__main__":
    main()