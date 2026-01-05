import os
import shutil
from ultralytics import RTDETR

# Configs
DATA_YAML = 'dataset/data.yaml'
RTDETR_WEIGHTS = 'analysis_temp/redetr/redetr/训练出来的模型/rtdetr-l_best.pt'
RTDETR_DIR = 'analysis_temp/redetr/redetr/'

def safe_copy(src, dst):
    if os.path.exists(src):
        try:
            shutil.copy(src, dst)
        except Exception as e:
            print(f"Error copying {src}: {e}")

def run_val():
    print(f"Starting RTDETR validation in background...")
    
    try:
        model = RTDETR(RTDETR_WEIGHTS)
        # Use a specific project directory
        project_dir = os.path.join(RTDETR_DIR, 'eval_runs')
        name = 'rtdetr_eval_bg'
        
        # Run validation
        metrics = model.val(data=DATA_YAML, project=project_dir, name=name, split='test', batch=4)
        
        print(f"RTDETR Validation Finished. mAP50: {metrics.box.map50}")
        
        # Copy plots
        run_dir = os.path.join(project_dir, name)
        
        # Copy plots to output directory
        for plot in ['confusion_matrix.png', 'F1_curve.png', 'PR_curve.png', 'P_curve.png', 'R_curve.png', 'confusion_matrix_normalized.png']:
            safe_copy(os.path.join(run_dir, plot), os.path.join(RTDETR_DIR, plot))
            
    except Exception as e:
        print(f"RTDETR Validation Failed: {e}")

if __name__ == '__main__':
    run_val()
