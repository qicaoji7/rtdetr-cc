import os
import sys
import torch
from ultralytics import YOLO

# 1. 设置环境变量
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_yaml = os.path.join(project_root, 'my_pipeline', 'city_damage.yaml')

def fast_yolo_tune():
    print(f"\n{'='*20} 启动 YOLOv8n 极速微调 {'='*20}")
    
    # 加载最轻量的模型
    model = YOLO('yolov8n.pt')

    # 训练配置：减少步数，增加 batch，开启快速模式
    results = model.train(
        data=data_yaml,
        epochs=10,         # 只跑 10 轮，微调足够了
        batch=16,          # 增加 batch size 利用 GPU
        imgsz=640,
        freeze=10,         # 冻结前 10 层 (Backbone)
        device='mps',      # 明确指定使用 Mac GPU
        project='my_pipeline/runs',
        name='yolov8n_quick_tune',
        exist_ok=True,
        workers=0          # Mac 下建议设置为 0 避免多进程卡死
    )
    
    print(f"\n{'='*20} 微调完成！{'='*20}")
    print(f"指标图片已生成在: my_pipeline/runs/yolov8n_quick_tune/results.png")

if __name__ == "__main__":
    fast_yolo_tune()

