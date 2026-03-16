import cv2
import numpy as np
"""透视矫正模块，解决刻度尺倾斜的问题，只服务于刻度尺的读数"""

class PerspectiveCorrector:
    def __init__(self):
        pass

    def correct_ruler(self, ruler_img):
        """预处理：灰度+高斯模糊+Canny边缘检测"""
        gray = cv2.cvtColor(ruler_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        """以下是光学实验图像预处理的标准做法，论文里经常有"""
        # 找最大轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return ruler_img, False
        max_contour = max(contours, key=cv2.contourArea)

        # 拟合四边形
        epsilon = 0.02 * cv2.arcLength(max_contour, True)
        approx = cv2.approxPolyDP(max_contour, epsilon, True)
        if len(approx) != 4:
            return ruler_img, False

        # 整理四个角点顺序
        pts = approx.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # 左上
        rect[2] = pts[np.argmax(s)]  # 右下
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # 右上
        rect[3] = pts[np.argmax(diff)]  # 左下
        """OPenCV经典透视矫正"""

        # 计算矫正后尺寸
        (tl, tr, br, bl) = rect
        maxWidth = max(int(np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))),
                       int(np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))))
        maxHeight = max(int(np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))),
                        int(np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))))

        # 透视变换
        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        corrected_img = cv2.warpPerspective(ruler_img, M, (maxWidth, maxHeight))

        return corrected_img, True