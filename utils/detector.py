from ultralytics import YOLO
import cv2
import numpy as np
import config#配置与核心逻辑解耦，我把硬参数都放在了全局配置文件config.py中

class FringeDetector:
    def __init__(self, model_path=config.MODEL_PATH):
        try:
            self.model = YOLO(model_path)
            self.class_dict = config.CLASS_DICT
            """config.CLASS_DICT需要和data.yaml里的类别、顺序完全一致！！！"""
            self.device = config.FORCE_DEVICE#我电脑没有GPU\CUDA,强制用CPU
        except Exception as e:
            raise RuntimeError(f"模型加载失败，请检查权重文件：{e}")

    def detect(self, img_path_or_array):
        #用opencv读图,提升了鲁棒性，兼容图片和数组
        if isinstance(img_path_or_array, str):
            img = cv2.imdecode(np.fromfile(img_path_or_array,dtype=np.uint8),cv2.IMREAD_COLOR)
            """兼容中文路径"""
        else:
            img = img_path_or_array.copy()

        if img is None:
            raise ValueError("图片读取失败，请检查路径或图片格式")

        results = self.model(
            img,
            conf=config.INFER_CONF_THRES,#置信度阈值，高于才会保留
            imgsz=config.INFER_IMGSZ,
            device=self.device,#没GPU，强制用CPU
            verbose = False #关掉日志，避免控制台废话太多
        )

        detect_result = {
            "original_img": img,
            "img_shape": img.shape,
            "detections": {}
        }

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0].cpu().numpy())
                class_name = self.class_dict[cls_id]
                confidence = float(box.conf[0].cpu().numpy())
                xyxy = box.xyxy[0].cpu().numpy()
                #box的下一级都是Pytorch的张量，.cpu().numpy()转化成数组
                x1, y1, x2, y2 = xyxy
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) /2

                detect_result["detections"][class_name] = {
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox_xyxy": xyxy,
                    # box.xyxy转化为int原因：OpenCV的绘图函数只认整数坐标
                    "bbox_xyxy_int": np.array(xyxy, dtype=int),
                    "center_pixel": (center_x, center_y)
                }

        return detect_result