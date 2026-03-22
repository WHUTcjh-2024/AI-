from ultralytics import YOLO
import config
import os

def train_yolo_cpu():
    print("="*60)
    print("YOLOv11-nano衍射条纹检测模型")
    print(f"运行设备：{config.FORCE_DEVICE} | CPU：AMD Ryzen7 8745H")
    print("="*60)

    if not os.path.exists(config.DATA_YAML_PATH):
        print(f"错误：数据集配置文件 {config.DATA_YAML_PATH} 不存在")
        print("请先创建data/data.yaml文件，并准备好标注好的数据集")
        return

    print("正在加载YOLOv11-nano预训练模型...")
    model = YOLO("yolo11n.pt")
    print("开始训练...")
    results = model.train(
        data=config.DATA_YAML_PATH,
        epochs=120,
        imgsz=config.INFER_IMGSZ,
        batch=4,
        device=config.FORCE_DEVICE,  # 强制CPU训练
        project=os.path.join(config.BASE_DIR, "models"),
        name="yolo11_cpu_train",
        patience=25,
        augment=True,
        degrees=15,
        perspective=0.001,
        flipud=0.2,
        fliplr=0.2,
        mosaic=0.3,
        mixup=0.1,
        workers=config.INFER_WORKERS,
        cache=False,
        rect=False,
        amp=False,
        lr0 = 0.001
    )

    print("="*60)
    print("训练完成！")
    print(f"最优模型权重路径：models/yolo11_cpu_train/weights/best.pt")
    print("请将best.pt文件复制到 models/ 根目录下，即可开始推理")
    print("="*60)

if __name__ == "__main__":
    train_yolo_cpu()