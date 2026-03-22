import os

#基础路径配置，可以获取config.py的绝对路径和根目录，动态拼接
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
DATA_YAML_PATH = os.path.join(BASE_DIR, "data.yaml")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FORCE_DEVICE = "cpu"
INFER_WORKERS = 2

#和模型训练时的data.yaml必须对齐！！！！！！！！！
CLASS_DICT = {
    0: "CentralFringe",
    1: "FirstFringe",
    2: "Ruler",
}
CLASS_NAMES = list(CLASS_DICT.values())
NUM_CLASSES = len(CLASS_NAMES)

INFER_CONF_THRES = 0.6#置信度阈值
INFER_IMGSZ = 640#必须和训练模型时的imgsz参数完全一致

OCR_CONF_THRES = 0.8    # 数字识别置信度阈值（用于过滤低质量识别结果）
SCALE_UNIT = "mm"

#条纹亚像素定位参数
FITTING_WINDOW_HALF_SIZE = 30
CLAHE_CLIP_LIMIT = 2.0#对比度增强的强度，图像增强算法
CLAHE_GRID_SIZE = (8, 8)# 直方图均衡化的网格大小

# ===================== 灰度图绘制配置 =====================
GRAY_PROFILE_HALF_HEIGHT = 10
GRAY_PLOT_DPI = 150
GRAY_PLOT_STYLE = "seaborn-v0_8"

# config.py 新增，修复tesseract的路径，防止跨平台崩溃
TESSERACT_CMD = os.getenv("TESSERACT_CMD", r'C:\Program Files\Tesseract-OCR\tesseract.exe')