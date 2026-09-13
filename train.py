# -*- coding: utf-8 -*-
"""车辆计数模型训练脚本（路径 B：自定义训练）。

在 Roboflow 的 vehicle counting 数据集上微调 yolov8n。
训练完成后会在 runs/detect/train/weights/best.pt 生成自定义权重，
main.py 会自动优先加载该权重。
"""
from ultralytics import YOLO

# 数据集配置文件（已修正为本地绝对路径）
DATA_YAML = r"D:/视觉计数系统/Vehicle-counting.v11-roboflow-instant-2--eval-.yolov8/data.yaml"

if __name__ == "__main__":
    # 加载 YOLOv8n 预训练权重作为基础模型
    model = YOLO("yolov8n.pt")

    # 在自定义数据集上训练（准确率优化版：提升小目标召回 + 数据增强鲁棒性）
    model.train(
        data=DATA_YAML,       # 数据集配置
        epochs=80,            # 训练轮数（由 30 提升，配合早停防过拟合）
        imgsz=960,            # 输入图像尺寸（由 640 提升，改善摩托车/远处车辆等小目标）
        batch=16,             # 批次大小（CPU 或显存不足可调小）
        patience=15,          # 早停轮数：验证集不再提升则提前结束
        workers=0,            # Windows 上设为 0 避免多进程加载问题
        project="runs/detect",
        name="train",
        # ---- 数据增强参数（提升对光照/角度/近远的鲁棒性） ----
        fliplr=0.5,           # 水平翻转概率
        degrees=10,           # 随机旋转角度（模拟拍摄角度变化）
        translate=0.1,        # 随机平移
        scale=0.5,            # 随机缩放幅度（模拟远近距离）
        hsv_h=0.015,          # 色调扰动（模拟不同光照）
        hsv_s=0.7,            # 饱和度扰动
        hsv_v=0.4,            # 明度扰动（模拟明暗）
        mosaic=1.0,           # 马赛克拼接
        mixup=0.2,            # 混合样本
    )