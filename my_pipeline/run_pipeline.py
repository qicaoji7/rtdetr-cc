import sys
import os
import torch
from ultralytics import RTDETR, YOLO

# 1. 环境变量设置
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# 路径配置
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

def fast_finetune(model_type, weights, data_yaml, epochs=20, batch=8):
    print(f"\n{'='*20} 正在对 {model_type} 进行快速微调 {'='*20}")
    
    if model_type == 'RTDETR':
        # 加载 RT-DETR-R18
        model = RTDETR(weights)
        freeze_layers = 15 # 冻结 R18 Backbone
    else:
        # 加载 YOLOv8n
        model = YOLO(weights)
        freeze_layers = 10 # 冻结 Nano Backbone

    # 硬件检测
    if torch.cuda.is_available():
        device = '0'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    # 开始微调
    # 使用 freeze 参数冻结前 N 层
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        freeze=freeze_layers, 
        project='my_pipeline/runs',
        name=f'{model_type.lower()}_fast_tune',
        exist_ok=True,
        imgsz=640,
        device=device,
        save=True,
        plots=True # 自动生成验证曲线
    )
    return model

def main():
    data_yaml = os.path.join(project_root, 'my_pipeline', 'city_damage.yaml')
    
    # 1. 微调 RT-DETR-R18 (小模型)
    # 如果本地没有 rtdetr-r18.pt，代码会自动从官方下载
    rtdetr_small_weights = os.path.join(project_root, 'weight/weights/rtdetr-r18.pt')
    if not os.path.exists(rtdetr_small_weights):
        rtdetr_small_weights = 'rtdetr-r18.pt'
    
    rtdetr_model = fast_finetune('RTDETR', rtdetr_small_weights, data_yaml, epochs=20)

    # 2. 微调 YOLOv8n (极小模型)
    yolo_small_weights = 'yolov8n.pt'
    yolo_model = fast_finetune('YOLOv8', yolo_small_weights, data_yaml, epochs=20)

    print(f"\n{'='*20} 快速微调任务完成 {'='*20}")
    print("结果查看指引：")
    print("1. 训练曲线与指标: 查看 my_pipeline/runs/*/results.png")
    print("2. 模型权重: 查看 my_pipeline/runs/*/weights/best.pt")
    print("3. 验证图片: 查看 my_pipeline/runs/*/val_batch0_labels.jpg")

if __name__ == "__main__":
    main()