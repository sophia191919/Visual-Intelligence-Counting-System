# -*- coding: utf-8 -*-
"""视觉智能计数系统主程序（车辆 / 物体计数）。

五步架构（纯手写代码，不调用 plot()）：
  1. 输入模块       —— 遍历输入图片文件夹并读取图片
  2. 检测模块       —— 加载权重并执行检测（优先 best.pt，缺省回退 yolov8n.pt）
  3. 类别筛选模块   —— 只保留车辆类别框
  4. 计数模块       —— 统计目标框总数
  5. 画框叠数·保存  —— cv2.rectangle 画框 + cv2.putText 叠数 + cv2.imwrite 保存到 output
"""
import os
import cv2
from ultralytics import YOLO

# ==================== 计数规则（写在注释里） ====================
# 【计数规则说明】
# 1. 数哪一类：计数对象是“物体”——本程序统计数据集中全部 5 类车辆：
#    bicycle（自行车）、bus（公交车）、car（汽车）、motorcycle（摩托车）、truck（卡车）。
#    任何一类都不能漏掉，以满足“计数系统对物体整体计数”的需求。
# 2. 阈值多少：采用“按类别置信度阈值”。
#    - 默认阈值 conf = 0.35（bus / car / truck 等一般类别）
#    - 摩托车 motorcycle、自行车 bicycle 单独放宽到 0.20
#    为什么：小目标（摩托 / 自行车）因像素少、特征弱，模型给出的置信度普遍偏低，
#    若统一用 0.35 会大量漏检；单独放宽阈值可在“提升召回”与“控制误检”之间平衡。
# 3. 为什么这样计数：任务要求“对物体整体计数”。检测时先用极低阈值(MIN_CONF)取回全部候选框，
#    再按上面的类别阈值精细过滤，最后统计框总数；配合 TTA 测试时增强与图像预处理进一步提升召回。
# ============================================================

# 车道 / 路径
BASE_DIR = r"D:/视觉计数系统"
DATASET_DIR = os.path.join(BASE_DIR, "Vehicle-counting.v11-roboflow-instant-2--eval-.yolov8")
INPUT_DIR = os.path.join(DATASET_DIR, "test", "images")   # 输入：待检测图片文件夹
OUTPUT_DIR = os.path.join(BASE_DIR, "output")              # 输出：结果图保存文件夹

# 需要计数的车辆类别（按名称匹配，兼容自定义模型与 COCO 预训练模型）
VEHICLE_NAMES = {"bicycle", "bus", "car", "motorcycle", "truck"}

DEFAULT_CONF = 0.35        # 默认置信度阈值（bus / car / truck 等一般类别）
IOU_THRESHOLD = 0.5        # NMS 去重阈值
MIN_CONF = 0.05            # 检测时先取全部候选框的最低阈值，再按类别阈值精细过滤
TTA_ENABLED = True         # TTA 测试时增强：多尺度+翻转融合，提升召回（更慢，CPU 卡顿可改 False）

# 按类别置信度阈值：小目标/易漏检类别（摩托车、自行车）单独放宽，提升召回
CLASS_CONF_THRESHOLD = {
    "motorcycle": 0.20,
    "bicycle": 0.20,
}

# 权重路径：优先使用训练出的自定义权重，否则回退官方 yolov8n.pt
BEST_PT = os.path.join(BASE_DIR, "runs", "detect", "train", "weights", "best.pt")
MODEL_PATH = BEST_PT if os.path.exists(BEST_PT) else "yolov8n.pt"


def load_model():
    """2. 检测模块：加载模型权重。"""
    print(f"[检测模块] 加载权重: {MODEL_PATH}")
    return YOLO(MODEL_PATH)


def preprocess_image(image):
    """图像预处理：小图自适应放大 + CLAHE 对比度增强。

    目的：低分辨率 / 光照不足 / 低对比度是漏检的主要来源，送模型前先改善这些因素。
    """
    h, w = image.shape[:2]
    # 1. 小图自适应放大：短边不足 640 时上采样，提升小目标（摩托/远处车辆）的有效像素
    min_side = 640
    if max(h, w) < min_side:
        scale = min_side / max(h, w)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # 2. CLAHE 自适应直方图均衡：提升低对比度 / 光照不足图片的细节
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    image = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    return image


def detect_vehicles(model, image):
    """2~3. 检测 + 类别筛选模块：返回筛选后的车辆检测框。

    流程：预处理 → TTA 检测（低阈值取全部候选）→ 按类别阈值过滤。
    返回每个元素为 (class_name, x1, y1, x2, y2)。
    """
    # 2. 检测模块：不调用 plot()，直接取原始 boxes 数据
    #    conf 用 MIN_CONF 拿到全部候选框，之后按“类别阈值”精细过滤；
    #    augment=TTA_ENABLED 开启测试时增强（多尺度/翻转融合，提升召回）
    image = preprocess_image(image)
    results = model.predict(
        image, conf=MIN_CONF, iou=IOU_THRESHOLD,
        verbose=False, augment=TTA_ENABLED,
    )

    detections = []
    for r in results:
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            continue
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()
        names = model.names

        for cls_id, conf, box in zip(cls_ids, confs, xyxy):
            name = names.get(int(cls_id), str(cls_id))
            # 3. 类别筛选模块：只保留 5 类车辆
            if name not in VEHICLE_NAMES:
                continue
            # 按类别阈值过滤：小目标（摩托/自行车）放宽，大目标用默认阈值
            threshold = CLASS_CONF_THRESHOLD.get(name, DEFAULT_CONF)
            if conf < threshold:
                continue
            x1, y1, x2, y2 = map(int, box)
            detections.append((name, x1, y1, x2, y2))
    return detections


def draw_and_save(image, detections, filename):
    """4. 计数模块 + 5. 画框叠数·保存模块。"""
    # 4. 计数模块：统计筛选后框的总数
    total_count = len(detections)
    print(f"图片 {filename} 中，车辆总数量为: {total_count}")

    # 5.1 画框：遍历目标框，用 OpenCV 手动画矩形并写类别标签（不用 plot()）
    for name, x1, y1, x2, y2 in detections:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 5.2 叠数：在左上角用红底白字叠加总数
    total_text = f"Total Vehicles: {total_count}"
    cv2.rectangle(image, (8, 8), (8 + 260, 8 + 42), (0, 0, 255), -1)
    cv2.putText(image, total_text, (16, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    # 5.3 保存：把画好框的结果图写入 output 文件夹
    save_path = os.path.join(OUTPUT_DIR, f"result_{filename}")
    cv2.imwrite(save_path, image)
    print(f"检测结果已保存至: {save_path}")
    return total_count


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = load_model()

    # 1. 输入模块：遍历输入图片文件夹
    image_files = sorted(
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    print(f"[输入模块] 待处理图片数: {len(image_files)}")

    grand_total = 0
    for filename in image_files:
        img_path = os.path.join(INPUT_DIR, filename)
        image = cv2.imread(img_path)
        if image is None:
            print(f"[输入模块] 跳过无法读取的图片: {filename}")
            continue

        detections = detect_vehicles(model, image)
        grand_total += draw_and_save(image, detections, filename)

    print(f"\n[完成] 共 {len(image_files)} 张图片，累计车辆总数: {grand_total}")
    print(f"[完成] 结果图已保存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()