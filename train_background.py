import os
import shutil
from ultralytics import RTDETR, YOLO

DATA_YAML = 'dataset/data.yaml'
# Use the best weights found so far as starting point
RTDETR_WEIGHTS = 'analysis_temp/redetr/redetr/训练出来的模型/rtdetr-l_best.pt'
YOLO_WEIGHTS = 'analysis_temp/yolo8n/yolo8n/训练出来的yolo8n模型/best.pt'

RTDETR_DIR = 'analysis_temp/redetr/redetr/'
YOLO_DIR = 'analysis_temp/yolo8n/yolo8n/'

def safe_copy(src, dst):
    if os.path.exists(src):
        try:
            shutil.copy(src, dst)
        except Exception as e:
            print(f"Error copying {src}: {e}")

def train_and_eval(model_type, weights, out_dir, name):
    print(f"Starting training for {model_type}...")
    project_dir = os.path.join(out_dir, 'train_runs')
    
    try:
        if model_type == 'RTDETR':
            model = RTDETR(weights)
            batch = 4
        else:
            model = YOLO(weights)
            batch = 16

        # Train
        # patience=10 to stop early if no improvement
        model.train(data=DATA_YAML, epochs=50, imgsz=640, project=project_dir, name=name, batch=batch, patience=10)
        
        # Best weights
        best_pt = os.path.join(project_dir, name, 'weights', 'best.pt')
        
        # Validate best
        if os.path.exists(best_pt):
            print(f"Validating best {model_type}...")
            # Re-load to ensure clean state
            if model_type == 'RTDETR':
                model = RTDETR(best_pt)
            else:
                model = YOLO(best_pt)
            
            model.val(data=DATA_YAML, project=project_dir, name=name+'_val', split='test')
            
            # Copy plots
            run_dir = os.path.join(project_dir, name)
            val_dir = os.path.join(project_dir, name+'_val')
            
            # Training plots
            for plot in ['results.png', 'confusion_matrix.png', 'F1_curve.png', 'PR_curve.png', 'confusion_matrix_normalized.png']:
                safe_copy(os.path.join(run_dir, plot), os.path.join(out_dir, "train_" + plot))
                
            # Validation plots
            for plot in ['confusion_matrix.png', 'F1_curve.png', 'PR_curve.png', 'P_curve.png', 'R_curve.png', 'confusion_matrix_normalized.png']:
                safe_copy(os.path.join(val_dir, plot), os.path.join(out_dir, plot))
                
            print(f"Finished {model_type}.")
        else:
            print(f"Training finished but best.pt not found for {model_type}")
            
    except Exception as e:
        print(f"Error training {model_type}: {e}")

if __name__ == '__main__':
    # Train YOLO first (faster)
    train_and_eval('YOLO', YOLO_WEIGHTS, YOLO_DIR, 'yolo_finetune')
    # Train RTDETR
    train_and_eval('RTDETR', RTDETR_WEIGHTS, RTDETR_DIR, 'rtdetr_finetune')
