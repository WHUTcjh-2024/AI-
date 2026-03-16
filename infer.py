import os
import config
from utils.detector import FringeDetector
from utils.scale_calibrator import ScaleCalibrator
from utils.fringe_locator import FringeSubpixelLocator
from utils.physics_calc import PhysicsCalculator
from utils.visualizer import ResultVisualizer

def main(image_path, theoretical_value=None, save_result=True):
    print("正在初始化功能模块...")
    try:
        detector = FringeDetector()
        scale_calibrator = ScaleCalibrator()
        fringe_locator = FringeSubpixelLocator()
        physics_calc = PhysicsCalculator()
        visualizer = ResultVisualizer()
    except Exception as e:
        print(f"模块初始化失败：{e}")
        return

    # 提取图片名称，用于保存结果
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    print(f"正在处理图片：{img_name}")

    # ===================== 第一步：AI目标检测 =====================
    try:
        detect_result = detector.detect(image_path)
    except Exception as e:
        print(f"目标检测失败：{e}")
        return

    detections = detect_result["detections"]
    original_img = detect_result["original_img"]

    # 检查关键目标是否检测到
    required_classes = ["CentralFringe", "FirstFringe", "Ruler"]
    missing_classes = [cls for cls in required_classes if cls not in detections]
    if missing_classes:
        print(f"检测失败，未识别到关键目标：{missing_classes}")
        return
    print("目标检测完成，成功识别条纹与刻度尺")

    # ===================== 第二步：刻度尺标定 =====================
    print("正在进行刻度尺自动标定...")
    ruler_bbox = detections["Ruler"]["bbox_xyxy_int"]
    x1_r, y1_r, x2_r, y2_r = ruler_bbox
    ruler_roi = original_img[y1_r:y2_r, x1_r:x2_r]
    scale_ratio, scale_msg = scale_calibrator.get_scale_ratio(ruler_roi)
    print(f"📏 刻度尺标定结果：{scale_msg}")

    # ===================== 第三步：条纹亚像素级定位 =====================
    print("正在进行条纹亚像素精密定位...")
    # 中央亮纹定位
    central_rough = detections["CentralFringe"]["center_pixel"]
    central_fit = fringe_locator.locate_subpixel_center(original_img, central_rough)
    if central_fit["success"]:
        print(f"中央亮纹定位完成，拟合优度R²={central_fit['r_squared']:.4f}")
        central_center = central_fit["subpixel_center"]
    else:
        print(f"中央亮纹拟合失败，使用粗定位结果：{central_fit['error']}")
        central_center = central_rough

    # 一级条纹定位
    first_rough = detections["FirstFringe"]["center_pixel"]
    first_fit = fringe_locator.locate_subpixel_center(original_img, first_rough)
    if first_fit["success"]:
        print(f"一级条纹定位完成，拟合优度R²={first_fit['r_squared']:.4f}")
        first_center = first_fit["subpixel_center"]
    else:
        print(f"一级条纹拟合失败，使用粗定位结果：{first_fit['error']}")
        first_center = first_rough

    # ===================== 第四步：物理量计算 =====================
    spacing_result = physics_calc.calculate_fringe_spacing(
        central_center, first_center, scale_ratio
    )
    pixel_distance = spacing_result["pixel_spacing"]
    physical_distance = spacing_result.get("physical_distance_mm")
    status = spacing_result.get("status", "success")
    relative_error = None
    print("=" * 60)
    print(" 最终测量结果（横向条纹 → 纵向间距）")
    print(f"条纹像素间距（Y轴）：{pixel_distance:.4f} px")
    if physical_distance is not None:
        print(f" 0级与1级条纹物理间距：{physical_distance:.4f} mm")
    else:
        print(f" 标定失败，无法计算物理间距")
    if relative_error is not None:
        print(f" 与理论值相对误差：{relative_error:.4f} %")
    print("=" * 60)

    # ===================== 第五步：生成灰度图与可视化结果 =====================
    print("正在生成可视化结果与灰度分布图...")
    # 1. 生成标注结果图
    annotation_img = visualizer.draw_annotation_image(
        original_img, detections, central_center, first_center, physical_distance, scale_ratio
    )

    # 2. 生成全图灰度分布曲线（ImageJ样式，核心新增功能）
    fringe_center_y = (central_center[1] + first_center[1]) / 2  # 取条纹中心的y坐标
    pixel_axis, full_gray_profile = fringe_locator.get_full_gray_profile(original_img, fringe_center_y)

    # 3. 保存所有结果
    result_data = {
        "scale_ratio": scale_ratio,
        "central_center": central_center,
        "first_center": first_center,
        "pixel_distance": pixel_distance,
        "physical_distance_mm": physical_distance,
        "relative_error": relative_error,
        "central_fit_result": central_fit,
        "first_fit_result": first_fit
    }

    if save_result:
        result_folder, anno_path, gray_path, fit_path, data_path = visualizer.save_result_files(
            img_name,
            annotation_img,
            (pixel_axis, full_gray_profile),
            (central_fit, first_fit),
            result_data
        )
        print(f"所有结果已保存到：{result_folder}")
        print(f"   - 标注结果图：{anno_path}")
        print(f"   - 灰度分布曲线图：{gray_path}")
        print(f"   - 高斯拟合结果图：{fit_path}")
        print(f"   - 测量数据报告：{data_path}")

    # 返回所有结果，便于后续调用
    return {
        "result_data": result_data,
        "annotation_img": annotation_img,
        "gray_profile": (pixel_axis, full_gray_profile),
        "central_fit": central_fit,
        "first_fit": first_fit
    }

if __name__ == "__main__":
    # ===================== 测试配置 =====================
    # 请把这里改成你的测试图片路径
    TEST_IMAGE_PATH = "test.jpg"
    # 可选：输入你的理论间距值(mm)，用于计算相对误差，没有就填None
    THEORETICAL_SPACING = None

    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"测试图片 {TEST_IMAGE_PATH} 不存在，请检查路径")
    else:
        main(TEST_IMAGE_PATH, theoretical_value=THEORETICAL_SPACING)