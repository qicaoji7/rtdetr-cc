import os
import shutil
import re

def parse_metrics(file_path):
    metrics = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if 'mAP@0.5' in line and 'mAP@0.5:0.95' not in line:
                    val = re.search(r'[\d.]+', line.split('|')[2]).group()
                    metrics['mAP50'] = float(val) / 100 if '%' in line else float(val)
                elif 'mAP@0.5:0.95' in line:
                    val = re.search(r'[\d.]+', line.split('|')[2]).group()
                    metrics['mAP50-95'] = float(val) / 100 if '%' in line else float(val)
                elif 'Precision' in line:
                    val = re.search(r'[\d.]+', line.split('|')[2]).group()
                    metrics['Precision'] = float(val) / 100 if '%' in line else float(val)
                elif 'Recall' in line:
                    val = re.search(r'[\d.]+', line.split('|')[2]).group()
                    metrics['Recall'] = float(val) / 100 if '%' in line else float(val)
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    return metrics

def main():
    # 1. Parse Data
    yolo_path = "analysis_temp/yolo8n/yolo8n/测试数据的评估指标/总的指标markdown.txt"
    rtdetr_path = "analysis_temp/redetr/redetr/测试数据/总指标markdown.txt"
    
    yolo_data = parse_metrics(yolo_path)
    rtdetr_data = parse_metrics(rtdetr_path)
    
    print("Parsed Data:")
    print("YOLO:", yolo_data)
    print("RTDETR:", rtdetr_data)

    # 2. Update README.md
    readme_path = "README.md"
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace Table Data
        # | **RT-DETR-L** (Ours) | **0.852** | **0.794** | **0.824** | **0.581** | 32.8 | 85.2 |
        # | **YOLOv8n** (Baseline)| 0.789 | 0.741 | 0.765 | 0.512 | **3.2** | **8.1** |
        
        # RT-DETR Row
        r_p = f"{rtdetr_data.get('Precision', 0):.3f}"
        r_r = f"{rtdetr_data.get('Recall', 0):.3f}"
        r_map50 = f"{rtdetr_data.get('mAP50', 0):.3f}"
        r_map95 = f"{rtdetr_data.get('mAP50-95', 0):.3f}"
        
        content = re.sub(r'\|\s*\*\*RT-DETR-L\*\*\s*\(Ours\)\s*\|.*?\|', 
                         f'| **RT-DETR-L** (Ours) | **{r_p}** | **{r_r}** | **{r_map50}** | **{r_map95}** | 32.8 | ~80ms |', 
                         content)

        # YOLO Row
        y_p = f"{yolo_data.get('Precision', 0):.3f}"
        y_r = f"{yolo_data.get('Recall', 0):.3f}"
        y_map50 = f"{yolo_data.get('mAP50', 0):.3f}"
        y_map95 = f"{yolo_data.get('mAP50-95', 0):.3f}"
        
        content = re.sub(r'\|\s*\*\*YOLOv8n\*\*\s*\(Baseline\)\|.*?\|', 
                         f'| **YOLOv8n** (Baseline)| {y_p} | {y_r} | {y_map50} | {y_map95} | **3.2** | **~10ms** |', 
                         content)
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("README.md updated with real metrics.")

    # 3. Move Images for Display
    # Ensure target dirs exist
    os.makedirs("my_pipeline/runs/rtdetr_fast_tune", exist_ok=True)
    os.makedirs("my_pipeline/quick_check_results/inference_demo4", exist_ok=True)
    os.makedirs("my_pipeline/quick_check_results/heatmap_result", exist_ok=True)

    # Move Results Curve
    src_curve = "analysis_temp/redetr/redetr/训练日志/results.png"
    dst_curve = "my_pipeline/runs/rtdetr_fast_tune/results.png"
    if os.path.exists(src_curve):
        shutil.copy(src_curve, dst_curve)
        print(f"Copied curve to {dst_curve}")

    # Move Inference Demo (Pick one good one)
    src_demo_dir = "analysis_temp/redetr/redetr/预测的图片例子"
    dst_demo_path = "my_pipeline/quick_check_results/inference_demo4/FallenTrees_ia_400005492_png_jpg.rf.aecfdcb65e9c98e8c73c5073279447f7.jpg"
    
    if os.path.exists(src_demo_dir):
        images = os.listdir(src_demo_dir)
        if images:
            # Pick the first one for demo
            shutil.copy(os.path.join(src_demo_dir, images[0]), dst_demo_path)
            print(f"Copied demo image to {dst_demo_path}")

    # 4. Update POSTER_CONTENT.md
    poster_path = "POSTER_CONTENT.md"
    if os.path.exists(poster_path):
        with open(poster_path, 'r', encoding='utf-8') as f:
            p_content = f.read()
        
        # | **mAP@50** | **82.4%** 🏆 | 76.5% |
        # | **Precision** | **0.85** | 0.79 |
        
        p_content = re.sub(r'\|\s*\*\*mAP@50\*\*\s*\|.*?\|', 
                           f'| **mAP@50** | **{float(r_map50)*100:.1f}%** 🏆 | {float(y_map50)*100:.1f}% |', 
                           p_content)
        
        p_content = re.sub(r'\|\s*\*\*Precision\*\*\s*\|.*?\|', 
                           f'| **Precision** | **{r_p}** | {y_p} |', 
                           p_content)

        with open(poster_path, 'w', encoding='utf-8') as f:
            f.write(p_content)
        print("POSTER_CONTENT.md updated with real metrics.")

if __name__ == "__main__":
    main()
