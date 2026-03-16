"""前面所有模块的输出（检测结果 + 亚像素中心 + 物理
间距 + 拟合曲线）一键生成实验报告级可视化（标注图 + ImageJ 风
格灰度曲线 + 高斯拟合对比图 + 测量数据 txt），并按图片自
动建文件夹保存。"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import config

plt.style.use(config.GRAY_PLOT_STYLE)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class ResultVisualizer:
    def __init__(self):
        self.output_dir = config.OUTPUT_DIR

    def draw_annotation_image(self, original_img, detections, central_center, first_center, physical_distance,
                              scale_ratio):
        """绘制标注了条纹、刻度尺、间距的结果图"""
        res_img = original_img.copy()

        # 1. 绘制刻度尺框（黄色）
        if "Ruler" in detections:
            x1_r, y1_r, x2_r, y2_r = detections["Ruler"]["bbox_xyxy_int"]
            cv2.rectangle(res_img, (x1_r, y1_r), (x2_r, y2_r), (0, 255, 255), 3)
            cv2.putText(res_img, "刻度尺", (x1_r, y1_r - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # 2. 绘制中央亮纹（绿色）
        cx, cy = int(round(central_center[0])), int(round(central_center[1]))
        cv2.circle(res_img, (cx, cy), 12, (0, 255, 0), -1)
        cv2.putText(res_img, "0级中央亮纹", (cx + 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 3. 绘制一级条纹（红色）
        fx, fy = int(round(first_center[0])), int(round(first_center[1]))
        cv2.circle(res_img, (fx, fy), 12, (0, 0, 255), -1)
        cv2.putText(res_img, "1级衍射条纹", (fx + 20, fy), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # 4. 绘制间距连线与标注
        cv2.line(res_img, (cx, cy), (fx, fy), (255, 0, 0), 3)
        mid_x, mid_y = (cx + fx) // 2, (cy + fy) // 2
        if physical_distance is not None:
            cv2.putText(res_img, f"间距: {physical_distance:.4f} mm", (mid_x, mid_y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
        else:
            cv2.putText(res_img, f"像素间距: {abs(fy - cy):.1f} px", (mid_x, mid_y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)

        return res_img

    def draw_gray_profile_plot(self, pixel_axis, gray_profile, save_path=None, show_peaks=True):
        """
        绘制衍射条纹灰度分布曲线图（完全匹配ImageJ样式）
        :param pixel_axis: 横轴像素坐标
        :param gray_profile: 纵轴灰度值
        :param save_path: 图片保存路径
        :param show_peaks: 是否标注峰值
        :return: 绘制好的matplotlib图像对象/保存路径
        """
        fig, ax = plt.subplots(figsize=(10, 4), dpi=config.GRAY_PLOT_DPI)

        # 绘制灰度曲线
        ax.plot(pixel_axis, gray_profile, color='#1f77b4', linewidth=1.2, label='灰度值分布')

        # 图表设置，匹配ImageJ风格
        ax.set_title("衍射条纹灰度分布曲线", fontsize=12)
        ax.set_xlabel("像素距离 (Pixel)", fontsize=10)
        ax.set_ylabel("灰度值 (Gray Value)", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        plt.tight_layout()

        # 保存或返回
        if save_path:
            plt.savefig(save_path, dpi=config.GRAY_PLOT_DPI, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            return fig

    def draw_gaussian_fit_plot(self, central_fit, first_fit, save_path=None):
        """绘制单个条纹的高斯拟合对比图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=config.GRAY_PLOT_DPI)

        # 中央亮纹拟合图
        if central_fit["success"]:
            ax1.plot(central_fit["x_coords"], central_fit["intensity_profile"], 'o', markersize=3, label='实际光强',
                     alpha=0.6)
            ax1.plot(central_fit["x_coords"], central_fit["fit_curve"], '-', linewidth=2, label='高斯拟合曲线')
            ax1.set_title("中央亮纹高斯拟合", fontsize=10)
            ax1.set_xlabel("像素坐标", fontsize=9)
            ax1.set_ylabel("灰度值", fontsize=9)
            ax1.grid(True, linestyle='--', alpha=0.5)
            ax1.legend()
            ax1.text(0.05, 0.95, f"R²={central_fit['r_squared']:.4f}", transform=ax1.transAxes,
                     bbox=dict(facecolor='white', alpha=0.8))

        # 一级条纹拟合图
        if first_fit["success"]:
            ax2.plot(first_fit["x_coords"], first_fit["intensity_profile"], 'o', markersize=3, label='实际光强',
                     alpha=0.6)
            ax2.plot(first_fit["x_coords"], first_fit["fit_curve"], '-', linewidth=2, label='高斯拟合曲线')
            ax2.set_title("一级条纹高斯拟合", fontsize=10)
            ax2.set_xlabel("像素坐标", fontsize=9)
            ax2.set_ylabel("灰度值", fontsize=9)
            ax2.grid(True, linestyle='--', alpha=0.5)
            ax2.legend()
            ax2.text(0.05, 0.95, f"R²={first_fit['r_squared']:.4f}", transform=ax2.transAxes,
                     bbox=dict(facecolor='white', alpha=0.8))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=config.GRAY_PLOT_DPI, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            return fig

    def save_result_files(self, img_name, annotation_img, gray_plot, fit_plot, result_data):
        """批量保存所有结果文件到output目录"""
        # 创建本次结果的文件夹
        result_folder = os.path.join(self.output_dir, img_name)
        os.makedirs(result_folder, exist_ok=True)

        # 1. 保存标注图
        anno_path = os.path.join(result_folder, f"{img_name}_标注结果.jpg")
        cv2.imwrite(anno_path, annotation_img)

        # 2. 保存灰度分布图
        gray_path = os.path.join(result_folder, f"{img_name}_灰度分布曲线.jpg")
        self.draw_gray_profile_plot(gray_plot[0], gray_plot[1], save_path=gray_path)

        # 3. 保存拟合曲线图
        fit_path = os.path.join(result_folder, f"{img_name}_高斯拟合结果.jpg")
        self.draw_gaussian_fit_plot(fit_plot[0], fit_plot[1], save_path=fit_path)

        # 4. 保存测量数据为文本
        data_path = os.path.join(result_folder, f"{img_name}_测量结果.txt")
        with open(data_path, 'w', encoding='utf-8') as f:
            f.write("===== 毛细波衍射AI测量结果 =====\n")
            f.write(f"图片名称：{img_name}\n")
            f.write(f"标定转换系数：{result_data['scale_ratio']:.4f} px/mm\n")
            f.write(
                f"中央亮纹亚像素中心：x={result_data['central_center'][0]:.4f} px, y={result_data['central_center'][1]:.4f} px\n")
            f.write(
                f"一级条纹亚像素中心：x={result_data['first_center'][0]:.4f} px, y={result_data['first_center'][1]:.4f} px\n")
            f.write(f"条纹像素间距：{result_data['pixel_distance']:.4f} px\n")
            f.write(f"条纹物理间距：{result_data['physical_distance_mm']:.4f} mm\n")
            if result_data.get('relative_error') is not None:
                f.write(f"与理论值相对误差：{result_data['relative_error']:.4f} %\n")
            f.write("\n===== 拟合精度报告 =====\n")
            f.write(f"中央亮纹拟合优度R²：{result_data['central_fit_result']['r_squared']:.4f}\n" if
                    result_data['central_fit_result']['success'] else "中央亮纹拟合失败\n")
            f.write(f"一级条纹拟合优度R²：{result_data['first_fit_result']['r_squared']:.4f}\n" if
                    result_data['first_fit_result']['success'] else "一级条纹拟合失败\n")

        return result_folder, anno_path, gray_path, fit_path, data_path