# -*- coding: utf-8 -*-
"""单张图片模型准确性检验脚本。

用法：
    python test_single.py <图片文件路径> [输出图片路径]

示例（必须是具体图片文件，不能是文件夹）：
    python test_single.py "D:/视觉计数系统/Vehicle-counting.v11-roboflow-instant-2--eval-.yolov8/test/images/001_jpg.rf.ae1142540690b374c96c0031a8e7d4f2.jpg"

说明：
    - 不需要提前准备 best.pt——没有 best.pt 时会自动回退官方 yolov8n.pt（COCO 预训练）。
    - 路径支持中文，本脚本用 np.fromfile + cv2.imdecode 读取，避免 cv2.imread 中文路径失败。
    - labels 文件夹里是 .txt 标注文件，不是图片，不能作为输入。
"""
import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO

# ==================== 计数规则（与 main.py 一致） ====================
# 数哪一类：bicycle / bus / car / motorcycle / truck 五类车辆，计数对象是“物体”。
# 阈值：conf = 0.35，在防漏检与过滤背景误检之间取平衡。
# ============================================================

BASE_DIR = r"D:/视觉计数系统"
VEHICLE_NAMES = {"bicycle", "bus", "car", "motorcycle", "truck"}
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.5
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# 自动查找图片时的候选目录（只传文件名时可在这里按名字匹配）
DATA_ROOT = os.path.join(BASE_DIR, "Vehicle-counting.v11-roboflow-instant-2--eval-.yolov8")
IMAGE_SEARCH_DIRS = [
    os.path.join(DATA_ROOT, "test", "images"),
    os.path.join(DATA_ROOT, "train", "images"),
    os.path.join(DATA_ROOT, "valid", "images"),
]

# 权重路径：优先 best.pt，否则回退 yolov8n.pt
BEST_PT = os.path.join(BASE_DIR, "runs", "detect", "train", "weights", "best.pt")
MODEL_PATH = BEST_PT if os.path.exists(BEST_PT) else "yolov8n.pt"


def imread_unicode(path):
    """读取图片（兼容中文路径）。cv2.imread 在 Windows 下可能无法读取中文路径，
    改用 np.fromfile 读字节流 + cv2.imdecode 解码。"""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img):
    """保存图片（兼容中文路径）。"""
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(path)
    return True


def list_images(folder):
    """列出文件夹内所有图片文件。"""
    return sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(IMAGE_EXTS)
    )


def resolve_image_path(path):
    """当只传了文件名、或相对路径在当前目录找不到时，自动到数据集图片目录里按文件名查找。

    返回唯一匹配的完整路径；找不到或同名多个时返回 None。
    """
    base = os.path.basename(path)
    if not base or not base.lower().endswith(IMAGE_EXTS):
        return None

    # 直接存在（含相对当前目录）则优先使用
    if os.path.exists(path):
        return path

    matches = []
    for d in IMAGE_SEARCH_DIRS:
        cand = os.path.join(d, base)
        if os.path.exists(cand):
            matches.append(cand)

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print("[错误] 在多个目录中找到同名文件，请改用完整路径：")
        for m in matches:
            print("  " + m)
        return None
    return None


def main():
    if len(sys.argv) < 2:
        print("用法: python test_single.py <图片文件路径> [输出图片路径]")
        sys.exit(1)

    img_path = sys.argv[1]

    # —— 处理“误传文件夹”的情况 ——
    if os.path.isdir(img_path):
        imgs = list_images(img_path)
        print(f"[错误] 你传的是文件夹，不是图片文件: {img_path}")
        print(f"       该文件夹内共有 {len(imgs)} 张图片。")
        if imgs:
            print("       请把命令中的路径补全到具体的 .jpg 文件，例如：")
            for name in imgs[:3]:
                print(f'         python test_single.py "{os.path.join(img_path, name)}"')
        else:
            print("       该文件夹内没有 .jpg/.png 图片（labels 文件夹里是 .txt 标注，不是图片）。")
        sys.exit(1)

    # —— 只传了文件名/相对路径时，自动到数据集图片目录查找 ——
    if not os.path.exists(img_path):
        resolved = resolve_image_path(img_path)
        if resolved is None:
            print(f"[错误] 找不到图片: {img_path}")
            print("       请确认文件名是否正确，或用完整路径，例如：")
            demo_dir = IMAGE_SEARCH_DIRS[0]
            for name in list_images(demo_dir)[:3]:
                print(f'         python test_single.py "{os.path.join(demo_dir, name)}"')
            sys.exit(1)
        print(f"[提示] 已自动匹配到图片: {resolved}")
        img_path = resolved

    image = imread_unicode(img_path)
    if image is None:
        print(f"[错误] 无法读取图片: {img_path}")
        print("       请确认文件是 .jpg/.png 等图片格式。")
        sys.exit(1)

    # 未指定输出路径时，自动在图片旁生成 result_ 前缀的结果图
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(img_path) or ".",
        "result_" + os.path.basename(img_path),
    )

    print(f"[检测模块] 加载权重: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # 2~3. 检测 + 类别筛选
    results = model.predict(image, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
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
            if name in VEHICLE_NAMES:
                x1, y1, x2, y2 = map(int, box)
                detections.append((name, float(conf), x1, y1, x2, y2))

    # 4. 计数模块：打印总数 + 每个目标的类别与置信度
    total = len(detections)
    print(f"[计数模块] 该图车辆总数: {total}")
    for name, conf, *_ in detections:
        print(f"  - {name:<10} conf={conf:.2f}")

    # 5. 画框叠数·保存
    for name, conf, x1, y1, x2, y2 in detections:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{name} {conf:.2f}"
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    total_text = f"Total Vehicles: {total}"
    cv2.rectangle(image, (8, 8), (8 + 260, 8 + 42), (0, 0, 255), -1)
    cv2.putText(image, total_text, (16, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    if imwrite_unicode(out_path, image):
        print(f"[画框叠数·保存] 结果图已保存至: {out_path}")
    else:
        print(f"[画框叠数·保存] 保存失败: {out_path}")


if __name__ == "__main__":
    main()