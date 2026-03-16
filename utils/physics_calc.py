"""物理量计算模块：把前面所有模块的输出转换成真实物理量，
    并计算相对误差"""

class PhysicsCalculator:
    def __init__(self):
        pass

    def calculate_fringe_spacing(self, center0_subpixel, center1_subpixel, scale_ratio):
        """计算0级和1级条纹的像素间距、物理间距(mm)
        """
        pixel_distance = abs(center1_subpixel[1] - center0_subpixel[1])  # ← 这里改成 [1] (y坐标)

        if scale_ratio is None or scale_ratio <= 0:
            physical_distance_mm = None
            status = "标定失败"
        else:
            physical_distance_mm = pixel_distance / scale_ratio
            status = "success"

        return {
            "pixel_spacing": pixel_distance,  # 像素间距（纵向）
            "physical_spacing_mm": physical_distance_mm,
            "status": status,
            "direction": "vertical"
        }

    def calculate_relative_error(self, measured_value, theoretical_value):
        """计算与理论值的相对误差"""
        if theoretical_value == 0:
            return None
        return abs(measured_value - theoretical_value) / theoretical_value * 100