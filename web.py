# -*- coding: utf-8 -*-
"""视觉智能计数系统 —— 图像上传与物体鉴别 Web 服务（FastAPI）。

启动方式：
    python web.py
然后在浏览器打开 http://127.0.0.1:8000

职责：
    1. 前端静态页面托管（static/index.html）
    2. POST /api/detect  —— 接收多张图片，运行 YOLO 检测，返回每个物体的类别/置信度/归一化坐标与分类计数
    3. GET  /api/status  —— 返回当前加载的模型名
"""
import os
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

# ==================== 计数 / 检测规则（写在注释里） ====================
# 【检测规则说明】
# 1. 数哪一类：本界面是“物体鉴别”，计数对象是检测到的所有物体类别
#    （用本项目的 best.pt 时即 bicycle/bus/car/motorcycle/truck 五类车辆；
#      用 yolov8n.pt 兜底时即 COCO 的 80 类）。按类别名称分组统计，任何一类都不漏。
# 2. 阈值多少：置信度阈值 conf = 0.35。
#    为什么：测试图中目标较远/边缘模糊时置信度会下降，0.35 在防漏检与过滤误检间取平衡。
# 3. 为什么这样计数：任务要求“对物体整体计数”，保留所有类别框、过滤低置信度无效框后按类别累加。
# ============================================================

BASE_DIR = r"D:/视觉计数系统"
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# 权重路径：优先 best.pt，否则回退官方 yolov8n.pt（与 main.py / test_single.py 一致）
BEST_PT = os.path.join(BASE_DIR, "runs", "detect", "train", "weights", "best.pt")
MODEL_PATH = BEST_PT if os.path.exists(BEST_PT) else "yolov8n.pt"

CONF_THRESHOLD = 0.35   # 默认置信度阈值（一般类别 / 缺省）
IOU_THRESHOLD = 0.5     # NMS 去重阈值
MIN_CONF = 0.05         # 检测时先取全部候选框的最低阈值，再按类别阈值精细过滤
TTA_ENABLED = True      # TTA 测试时增强：提升召回（CPU 下会明显变慢，卡顿可改 False）

# 按类别置信度阈值：小目标/易漏检类别（摩托车、自行车）单独放宽，提升召回
CLASS_CONF_THRESHOLD = {
    "motorcycle": 0.20,
    "bicycle": 0.20,
}

# 启动即加载模型（避免每次请求重复加载）
model = YOLO(MODEL_PATH)
print(f"[检测模块] 加载权重: {MODEL_PATH}")


def preprocess_image(image):
    """图像预处理：小图自适应放大 + CLAHE 对比度增强。

    目的：低分辨率 / 光照不足 / 低对比度是漏检的主要来源，送模型前先改善这些因素。
    """
    h, w = image.shape[:2]
    # 1. 小图自适应放大：短边不足 640 时上采样，提升小目标的有效像素
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


app = FastAPI(title="视觉智能计数系统")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    """返回前端页面。"""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/status")
def status():
    """返回当前加载的模型名。"""
    return {"model": os.path.basename(MODEL_PATH)}


@app.post("/api/detect")
async def detect(files: list[UploadFile] = File(...), conf: float = CONF_THRESHOLD):
    """批量检测上传的图片。

    返回结构：
      {
        "model": "yolov8n.pt",
        "results": [
          {
            "filename": "xx.jpg",
            "error": null,
            "width": 640, "height": 480,
            "total": 5,
            "counts": [{"class": "car", "count": 3}, ...],   # 按数量降序
            "detections": [
               {"class": "car", "conf": 0.84, "box": [x1, y1, x2, y2]},  # 归一化坐标 0~1
               ...
            ]
          },
          ...
        ]
      }
    """
    payload = []
    for f in files:
        data = await f.read()
        # 从字节流解码图片（无中文路径问题）
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            payload.append({"filename": f.filename, "error": "无法识别的图片（非图片字节或已损坏）"})
            continue

        image = preprocess_image(image)
        h, w = image.shape[:2]
        # TTA 测试时增强 + 低阈值取全部候选框，之后按类别阈值精细过滤
        result = model.predict(
            image, conf=MIN_CONF, iou=IOU_THRESHOLD,
            verbose=False, augment=TTA_ENABLED,
        )[0]

        detections = []
        counts = {}
        boxes = getattr(result, "boxes", None)
        names = model.names
        if boxes is not None and len(boxes) > 0:
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            xyxyn = boxes.xyxyn.cpu().numpy()  # 归一化 x1,y1,x2,y2 ∈ [0,1]
            for cls_id, cf, xy in zip(cls_ids, confs, xyxyn):
                name = names.get(int(cls_id), str(cls_id))
                threshold = CLASS_CONF_THRESHOLD.get(name, conf)
                if cf < threshold:
                    continue
                detections.append({
                    "class": name,
                    "conf": round(float(cf), 3),
                    "box": [round(float(v), 4) for v in xy],
                })
                counts[name] = counts.get(name, 0) + 1

        payload.append({
            "filename": f.filename,
            "error": None,
            "width": w,
            "height": h,
            "total": len(detections),
            "counts": [
                {"class": k, "count": v}
                for k, v in sorted(counts.items(), key=lambda kv: -kv[1])
            ],
            "detections": detections,
        })

    return {"model": os.path.basename(MODEL_PATH), "results": payload}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)