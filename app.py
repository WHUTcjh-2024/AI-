import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import sys

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from utils.detector import FringeDetector
from utils.scale_calibrator import ScaleCalibrator
from utils.fringe_locator import FringeSubpixelLocator
from utils.physics_calc import PhysicsCalculator
from utils.visualizer import ResultVisualizer

st.set_page_config(
    page_title="毛细波衍射精密测量系统",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ===================== 模型缓存 =====================
@st.cache_resource
def load_modules_cpu():
    """缓存加载所有模块，避免重复初始化"""
    try:
        detector = FringeDetector()
        scale_calibrator = ScaleCalibrator()
        fringe_locator = FringeSubpixelLocator()
        physics_calc = PhysicsCalculator()
        visualizer = ResultVisualizer()
        return detector, scale_calibrator, fringe_locator, physics_calc, visualizer, "success"
    except Exception as e:
        return None, None, None, None, None, str(e)

# ===================== 主界面逻辑 =====================
def main():
    st.title("🔬 AI+物理实验：毛细波衍射精密测量系统")
    st.markdown("**物理启发式亚像素级测量 | 全自动标定 | 衍射条纹灰度分布可视化**")
    st.divider()
    # 侧边栏
    with st.sidebar:
        st.header("📦 系统状态")
        with st.spinner("正在加载模块（CPU运行，首次加载约10秒）..."):
            detector, scale_calibrator, fringe_locator, physics_calc, visualizer, status = load_modules_cpu()

        if status == "success":
            st.success("✅ 所有模块加载完成")
        else:
            st.error(f"❌ 模块加载失败：{status}")
            st.info("💡 排查提示：请先运行train.py完成训练，将best.pt权重放入models文件夹")

        st.divider()
        st.header("⚙️ 实验参数设置")
        theoretical_spacing = st.number_input(
            "条纹间距理论值 (mm)",
            value=0.0,
            format="%.4f",
            help="输入理论计算的间距，自动计算相对误差"
        )
        st.divider()
        st.markdown("### 📖 使用说明")
        st.markdown("1. 上传包含衍射条纹和刻度尺的照片")
        st.markdown("2. 点击「开始AI分析」按钮")
        st.markdown("3. 自动完成测量、生成灰度图与结果报告")

    # 模块加载失败则终止
    if status != "success":
        st.stop()

    # 图片上传
    uploaded_file = st.file_uploader("📷 上传衍射图像（JPG/PNG格式）", type=["jpg", "jpeg", "png"])

    # 结果变量初始化
    analysis_result = None
    annotation_img_rgb = None
    gray_profile_data = None
    central_fit = None
    first_fit = None

    if uploaded_file is not None:
        # 读取图片
        image = Image.open(uploaded_file)
        img_rgb = np.array(image.convert("RGB"))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # 两栏布局展示原图和标注图
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("原始图像")
            st.image(image, use_container_width=True)

        # 分析按钮
        if st.button("🚀 开始AI分析", type="primary", use_container_width=True):
            with st.status("正在执行全流程分析（CPU运行中）...", expanded=True) as status_bar:
                # 1. 目标检测
                status_bar.update(label="第一步：AI定位条纹与刻度尺", state="running")
                try:
                    detect_result = detector.detect(img_bgr)
                    detections = detect_result["detections"]
                except Exception as e:
                    st.error(f"目标检测失败：{e}")
                    status_bar.update(label="分析失败", state="error")
                    st.stop()

                # 检查关键目标
                required_classes = ["CentralFringe", "FirstFringe", "Ruler"]
                missing = [cls for cls in required_classes if cls not in detections]
                if missing:
                    st.error(f"检测失败，未识别到关键目标：{missing}")
                    status_bar.update(label="分析失败", state="error")
                    st.stop()

                # 2. 刻度尺标定
                status_bar.update(label="第二步：刻度尺自动标定与畸变矫正", state="running")
                ruler_bbox = detections["Ruler"]["bbox_xyxy_int"]
                x1_r, y1_r, x2_r, y2_r = ruler_bbox
                ruler_roi = img_bgr[y1_r:y2_r, x1_r:x2_r]
                scale_ratio, scale_msg = scale_calibrator.get_scale_ratio(ruler_roi)
                if scale_ratio is None:
                    st.warning(f"刻度尺标定警告：{scale_msg}")

                # 3. 亚像素定位
                status_bar.update(label="第三步：衍射条纹亚像素级精密定位", state="running")
                central_rough = detections["CentralFringe"]["center_pixel"]
                central_fit = fringe_locator.locate_subpixel_center(img_bgr, central_rough)
                first_rough = detections["FirstFringe"]["center_pixel"]
                first_fit = fringe_locator.locate_subpixel_center(img_bgr, first_rough)

                central_center = central_fit["subpixel_center"]
                first_center = first_fit["subpixel_center"]

                # 4. 物理量计算
                status_bar.update(label="第四步：物理量计算与误差分析", state="running")
                pixel_dist, physical_dist = physics_calc.calculate_fringe_spacing(
                    central_center, first_center, scale_ratio
                )
                relative_error = None
                if theoretical_spacing > 0 and physical_dist is not None:
                    relative_error = physics_calc.calculate_relative_error(physical_dist, theoretical_spacing)

                # 5. 生成灰度分布曲线
                status_bar.update(label="第五步：生成灰度分布曲线与可视化结果", state="running")
                # 生成标注图
                annotation_img = visualizer.draw_annotation_image(
                    img_bgr, detections, central_center, first_center, physical_dist, scale_ratio
                )
                annotation_img_rgb = cv2.cvtColor(annotation_img, cv2.COLOR_BGR2RGB)
                # 生成全图灰度分布
                fringe_center_y = (central_center[1] + first_center[1]) / 2
                pixel_axis, full_gray_profile = fringe_locator.get_full_gray_profile(img_bgr, fringe_center_y)
                gray_profile_data = (pixel_axis, full_gray_profile)

                # 整理结果
                analysis_result = {
                    "scale_ratio": scale_ratio,
                    "scale_msg": scale_msg,
                    "central_center": central_center,
                    "first_center": first_center,
                    "pixel_distance": pixel_dist,
                    "physical_distance": physical_dist,
                    "relative_error": relative_error,
                    "central_fit": central_fit,
                    "first_fit": first_fit
                }
                status_bar.update(label="✅ 全流程分析完成！", state="complete")

        # 展示标注结果图
        if annotation_img_rgb is not None:
            with col2:
                st.subheader("AI分析标注结果")
                st.image(annotation_img_rgb, use_container_width=True)

        # 展示测量数据报告
        if analysis_result is not None:
            st.divider()
            st.subheader("📊 精密测量数据报告")
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric(
                    label="条纹物理间距",
                    value=f"{analysis_result['physical_distance']:.4f} mm" if analysis_result[
                                                                                  'physical_distance'] is not None else "无法计算"
                )
            with metric_cols[1]:
                st.metric(
                    label="亚像素级像素间距",
                    value=f"{analysis_result['pixel_distance']:.3f} px"
                )
            with metric_cols[2]:
                st.metric(
                    label="标定转换系数",
                    value=f"{analysis_result['scale_ratio']:.2f} px/mm" if analysis_result[
                                                                               'scale_ratio'] is not None else "标定失败"
                )
            with metric_cols[3]:
                if analysis_result['relative_error'] is not None:
                    st.metric(
                        label="与理论值相对误差",
                        value=f"{analysis_result['relative_error']:.2f} %"
                    )
                else:
                    st.metric(label="相对误差", value="未输入理论值")

            # 新增：灰度分布曲线展示（核心需求）
            st.divider()
            st.subheader("📈 衍射条纹灰度分布曲线（ImageJ样式）")
            if gray_profile_data is not None:
                pixel_axis, gray_profile = gray_profile_data
                # 用streamlit的线图展示，和matplotlib效果一致
                import pandas as pd
                gray_df = pd.DataFrame({
                    "像素距离 (Pixel)": pixel_axis,
                    "灰度值 (Gray Value)": gray_profile
                })
                st.line_chart(gray_df, x="像素距离 (Pixel)", y="灰度值 (Gray Value)", height=400,
                              use_container_width=True)
                st.caption("说明：曲线峰值对应衍射条纹的亮纹中心，横轴为水平方向像素坐标，纵轴为图像灰度值")

            # 高斯拟合结果展示
            st.divider()
            st.subheader("🔍 亚像素高斯拟合结果（物理先验验证）")
            fit_cols = st.columns(2)
            with fit_cols[0]:
                st.markdown("#### 中央亮纹光强分布与高斯拟合")
                if central_fit["success"]:
                    fit_df = pd.DataFrame({
                        "像素坐标": central_fit["x_coords"],
                        "实际光强": central_fit["intensity_profile"],
                        "高斯拟合曲线": central_fit["fit_curve"]
                    })
                    st.line_chart(fit_df, x="像素坐标", y=["实际光强", "高斯拟合曲线"], height=300)
                    st.caption(
                        f"拟合优度 R² = {central_fit['r_squared']:.4f} | 亚像素中心 x = {central_fit['subpixel_center'][0]:.3f} px")
                else:
                    st.warning("中央亮纹高斯拟合失败")

            with fit_cols[1]:
                st.markdown("#### 一级条纹光强分布与高斯拟合")
                if first_fit["success"]:
                    fit_df = pd.DataFrame({
                        "像素坐标": first_fit["x_coords"],
                        "实际光强": first_fit["intensity_profile"],
                        "高斯拟合曲线": first_fit["fit_curve"]
                    })
                    st.line_chart(fit_df, x="像素坐标", y=["实际光强", "高斯拟合曲线"], height=300)
                    st.caption(
                        f"拟合优度 R² = {first_fit['r_squared']:.4f} | 亚像素中心 x = {first_fit['subpixel_center'][0]:.3f} px")
                else:
                    st.warning("一级条纹高斯拟合失败")

if __name__ == "__main__":
    main()