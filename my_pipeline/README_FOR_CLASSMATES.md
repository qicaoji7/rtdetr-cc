# RT-DETR 城市损坏检测微调实验指南

## 1. 实验简介
本项目基于改进的 RT-DETR 模型，对城市损坏（如道路破损、落石、违章停车等 10 类问题）进行微调训练。

## 2. 环境配置
```bash
# 1. 安装项目源码
pip install -e .

# 2. 安装必备库
pip install timm==1.0.7 thop einops grad-cam==1.5.4 prettytable seaborn opencv-python
```

## 3. 数据配置
数据集路径：`dataset/merged`
配置文件：`my_pipeline/city_damage.yaml` (请根据实际情况修改文件内的 path 绝对路径)

## 4. 训练脚本 (微调)
运行 `my_pipeline/run_pipeline.py`。
注意：请将该脚本内的 `QUICK_RUN` 设置为 `False` 以开启全量训练。

## 5. 可视化分析
- **推理结果**：查看 `runs/city_damage/*/inference_demo`
- **热力图**：查看 `my_pipeline/quick_check_results/heatmap_result`
- **训练曲线**：查看 `runs/city_damage/*/results.png`
