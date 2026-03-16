import pytesseract
import cv2
import numpy as np
import config
from utils.perspective import PerspectiveCorrector
"""刻度尺标定模块，tesseract需要单独安装"""
pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD

class ScaleCalibrator:
    def __init__(self):
        self.perspective_corrector = PerspectiveCorrector()

    def preprocess_ruler_for_ocr(self, ruler_img):
        # 1. 转灰度,简化计算，去除颜色干扰
        gray = cv2.cvtColor(ruler_img, cv2.COLOR_BGR2GRAY)
        # 2.高斯自适应（解决光照不均）
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        # 3. 降噪（去除小噪点）
        kernel = np.ones((1, 1), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # 4. 轻微膨胀，增强数字边缘，提升OCR识别率
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        return thresh

    def get_scale_ratio(self, ruler_img):
        """输入刻度尺ROI，输出 像素/毫米 转换系数（基于Tesseract OCR）"""
        # 1. 透视矫正
        corrected_ruler, is_corrected = self.perspective_corrector.correct_ruler(ruler_img)
        #以下为我自己新加的，可以提高鲁棒性
        if not is_corrected:
            return None,"透视矫正失败，请手动拍摄正视图"

        # 2. 图像预处理，提升OCR精度
        ocr_img = self.preprocess_ruler_for_ocr(corrected_ruler)

        # 3. Tesseract OCR识别数字（仅识别数字，关闭其他字符）
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
        ocr_result = pytesseract.image_to_data(
            ocr_img,
            output_type=pytesseract.Output.DICT,
            config=custom_config
        )

        # 4. 解析OCR结果，过滤低置信度
        valid_digits = []
        for i in range(len(ocr_result['text'])):
            text = ocr_result['text'][i].strip()
            conf = int(ocr_result['conf'][i])
            if conf > config.OCR_CONF_THRES and text.isdigit():
                # 提取数字的边界框中心坐标
                x = ocr_result['left'][i] + ocr_result['width'][i] / 2
                y = ocr_result['top'][i] + ocr_result['height'][i] / 2
                valid_digits.append((int(text), x, y))

        if len(valid_digits) < 2:
            return None, f"有效数字不足（仅识别到{len(valid_digits)}个），无法完成标定"

        # 5. 判断刻度尺方向（水平/垂直）
        y_coords = [d[2] for d in valid_digits]
        y_var = np.var(y_coords)
        is_horizontal = y_var < 20

        # 6. 按数字大小排序，取首尾计算
        valid_digits.sort(key=lambda x: x[0])
        start_val, start_px, start_py = valid_digits[0]
        end_val, end_px, end_py = valid_digits[-1]

        # 7. 计算标定系数
        physical_distance_mm = abs(end_val - start_val) * 10  # 刻度数字单位为厘米，转毫米
        if is_horizontal:
            pixel_distance = abs(end_px - start_px)
        else:
            pixel_distance = abs(end_py - start_py)

        if physical_distance_mm <= 0 or pixel_distance <= 0:
            return None, "标定距离计算错误（数字坐标无效）"

        ratio = pixel_distance / physical_distance_mm  # 1毫米对应多少像素

        return ratio, f"标定成功：{physical_distance_mm}mm对应{pixel_distance:.1f}像素，比例{ratio:.2f}px/mm"