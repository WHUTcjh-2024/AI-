import cv2
import numpy as np
from scipy.optimize import curve_fit
import config
"""衍射条纹的亚像素级中心定位
将YOLO粗定位精度提升至亚像素级别（R²评估拟合质量），满足精密光学条纹分析需求。
衍射条纹光强的高斯拟合函数，物理先验建模，比简单质心法精度高5-10倍"""
def gaussian_func(x, amplitude, center, sigma, offset):
    return amplitude * np.exp(-((x - center) ** 2) / (2 * sigma ** 2)) + offset

class FringeSubpixelLocator:
    def __init__(self):
        # 自适应光照增强，解决光照不均
        self.clahe = cv2.createCLAHE(
            clipLimit=config.CLAHE_CLIP_LIMIT,
            tileGridSize=config.CLAHE_GRID_SIZE
        )
        self.fitting_half_window = config.FITTING_WINDOW_HALF_SIZE

    def locate_subpixel_center(self, img, rough_center):
        """输入原图和粗定位中心，输出亚像素级条纹中心"""
        # 预处理：灰度化+对比度增强
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        enhanced_gray = self.clahe.apply(gray)
        h, w = enhanced_gray.shape

        # 粗定位中心取整
        cx_rough, cy_rough = int(round(rough_center[0])), int(round(rough_center[1]))

        # 拟合窗口边界检查
        x_min = max(0, cx_rough - self.fitting_half_window)
        x_max = min(w - 1, cx_rough + self.fitting_half_window)
        y_min = max(0, cy_rough - 5)
        y_max = min(h - 1, cy_rough + 5)

        #多行平均降噪
        intensity_profile = np.mean(enhanced_gray[y_min:y_max, x_min:x_max], axis=0)
        x_coords = np.arange(x_min, x_max, 1)

        # 高斯拟合
        try:
            # 初始化拟合参数
            init_amplitude = np.max(intensity_profile) - np.min(intensity_profile)
            init_center = cx_rough

            #使用半高全宽（FWHM）的粗略估计来设置init_sigma
            mask = intensity_profile > (np.max(intensity_profile) - np.min(intensity_profile)) / 2
            init_sigma = np.sum(mask) / 2.355

            init_offset = np.min(intensity_profile)
            p0 = [init_amplitude, init_center, init_sigma, init_offset]

            # 非线性最小二乘拟合，传入 p0 初始猜值，执行拟合
            popt, pcov = curve_fit(gaussian_func, x_coords, intensity_profile, p0=p0)
            fit_amplitude, fit_center, fit_sigma, fit_offset = popt

            # 计算拟合优度
            residuals = intensity_profile - gaussian_func(x_coords, *popt)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((intensity_profile - np.mean(intensity_profile)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            return {
                "success": True,
                "subpixel_center": (fit_center, cy_rough),
                "fit_params": popt,
                "r_squared": r_squared,
                "x_coords": x_coords,
                "intensity_profile": intensity_profile,
                "fit_curve": gaussian_func(x_coords, *popt)
            }

        except Exception as e:
            # 拟合失败返回粗定位结果
            return {
                "success": False,
                "subpixel_center": rough_center,
                "error": f"高斯拟合失败: {str(e)}"
            }

    def get_full_gray_profile(self, img, fringe_center_y):
        """
        新增：获取全宽度的条纹灰度分布曲线（匹配ImageJ样式）
        输入原图和条纹中心y坐标，输出水平方向的平均灰度分布
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        enhanced_gray = self.clahe.apply(gray)
        h, w = enhanced_gray.shape

        # 取中心上下多行做平均，降低噪声
        y_half = config.GRAY_PROFILE_HALF_HEIGHT
        y_min = max(0, int(fringe_center_y) - y_half)
        y_max = min(h - 1, int(fringe_center_y) + y_half)

        # 水平方向全宽度的平均灰度值
        full_gray_profile = np.mean(enhanced_gray[y_min:y_max, :], axis=0)
        pixel_axis = np.arange(0, w, 1)  # 横轴：像素距离

        return pixel_axis, full_gray_profile