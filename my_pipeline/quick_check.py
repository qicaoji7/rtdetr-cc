import os
import sys
import shutil

# 1. 强制设置环境变量，解决 Mac MPS 算子缺失问题
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# 添加项目根目录到 path，以便导入 heatmap
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from ultralytics import RTDETR
# 尝试导入 heatmap 脚本中的类 (假设 heatmap.py 定义了 rtdetr_heatmap 类)
try:
    from heatmap import rtdetr_heatmap
except ImportError as e:
    print(f"Warning: Could not import heatmap module: {e}")
    rtdetr_heatmap = None

def quick_check():
    print(f"\n{'='*20} 开始极速验证 (不训练，仅推理) {'='*20}")
    
    output_dir = os.path.join(project_root, 'my_pipeline', 'quick_check_results')
    
    # 找一张测试图片
    test_images_dir = os.path.join(project_root, 'dataset/merged/images/test')
    images = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        print("错误：测试集中没有找到图片！")
        return
    target_img_path = os.path.join(test_images_dir, images[0])
    print(f"验证目标图片: {target_img_path}")

    # 模型权重
    weights = os.path.join(project_root, 'weight/weights/rtdetr-l.pt')
    if not os.path.exists(weights):
        weights = 'rtdetr-l.pt' # 会自动下载

    # ---------------------------------------------------------
    # 任务 1: 运行推理 (RT-DETR)
    # ---------------------------------------------------------
    print(f"\n---> [1/2] 正在运行 RT-DETR 推理...")
    try:
        model = RTDETR(weights)
        results = model.predict(
            source=target_img_path,
            project=output_dir,
            name='inference_demo',
            save=True,
            conf=0.25,
            device='mps' # 尝试使用 MPS，如果不支持会自动 fallback
        )
        print(f"推理成功！结果保存在: {output_dir}/inference_demo")
    except Exception as e:
        print(f"推理失败: {e}")

    # ---------------------------------------------------------
    # 任务 2: 生成热力图
    # ---------------------------------------------------------
    print(f"\n---> [2/2] 正在生成热力图...")
    
    if rtdetr_heatmap is None:
        print("跳过热力图生成 (无法导入模块)")
        return

    heatmap_output_dir = os.path.join(output_dir, 'heatmap_result')
    if os.path.exists(heatmap_output_dir):
        shutil.rmtree(heatmap_output_dir)
    os.makedirs(heatmap_output_dir, exist_ok=True)

    # 构造参数，覆盖 heatmap.py 中的默认参数
    heatmap_params = {
        'weight': weights,
        'device': 'cpu', # 热力图强制用 CPU 确保稳定
        'method': 'GradCAM', 
        'layer': [15, 19, 22, 25], # RT-DETR 常用层
        'backward_type': 'all',
        'conf_threshold': 0.2,
        'ratio': 1.0,
        'show_box': True,
        'renormalize': True
    }

    try:
        # 实例化热力图模型
        model_heatmap = rtdetr_heatmap(**heatmap_params)
        # 运行处理
        # heatmap.py 的 __call__ 接受 (img_path, save_path)
        # 注意：这里 save_path 最好是一个目录，因为脚本内部会处理文件名
        model_heatmap(target_img_path, heatmap_output_dir)
        print(f"热力图生成成功！请查看: {heatmap_output_dir}")
    except Exception as e:
        print(f"热力图生成失败: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*20} 验证结束 {'='*20}")

if __name__ == "__main__":
    quick_check()