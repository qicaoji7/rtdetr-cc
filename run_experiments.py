import os
import shutil
from ultralytics import RTDETR, YOLO

# Paths
DATA_YAML = 'dataset/data.yaml'
RTDETR_WEIGHTS = 'analysis_temp/redetr/redetr/训练出来的模型/rtdetr-l_best.pt'
YOLO_WEIGHTS = 'analysis_temp/yolo8n/yolo8n/训练出来的yolo8n模型/best.pt'

RTDETR_DIR = 'analysis_temp/redetr/redetr/'
YOLO_DIR = 'analysis_temp/yolo8n/yolo8n/'

def safe_copy(src, dst):
    if os.path.exists(src):
        print(f"Copying {src} to {dst}")
        shutil.copy(src, dst)
    else:
        print(f"File not found: {src}")

def process_model(model_type, weights, out_dir, name):
    print(f"\n{'='*20}\nProcessing {model_type} with weights {weights}...\n{'='*20}")
    
    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)
    
    # Initialize model
    try:
        if model_type == 'RTDETR':
            model = RTDETR(weights)
        else:
            model = YOLO(weights)
    except Exception as e:
        print(f"Failed to load model {weights}: {e}")
        return

    # Try validation first
    print("Attempting initial validation...")
    project_dir = os.path.join(out_dir, 'eval_runs')
    
    need_training = False
    
    try:
        # We use a specific project/name to easily locate results
        # Note: If classes mismatch, this might throw an error or return 0 mAP
        metrics = model.val(data=DATA_YAML, project=project_dir, name=name, split='test')
        
        # Check mAP50 (metrics.box.map50)
        map50 = metrics.box.map50
        print(f"Initial Validation mAP50: {map50}")

        # If mAP is very low, we assume class mismatch or poor transfer, so we train
        if map50 < 0.1:
            print("Metrics are low (< 0.1). Marking for training...")
            need_training = True
        else:
            print("Metrics are acceptable. Copying validation plots...")
            run_dir = os.path.join(project_dir, name)
            for plot in ['confusion_matrix.png', 'F1_curve.png', 'PR_curve.png', 'P_curve.png', 'R_curve.png', 'confusion_matrix_normalized.png']:
                safe_copy(os.path.join(run_dir, plot), os.path.join(out_dir, plot))

    except Exception as e:
        print(f"Initial validation failed (likely class mismatch): {e}")
        need_training = True

    if need_training:
        print("\nStarting training (fine-tuning) for 20 epochs...")
        try:
            # Re-init model to ensure clean state if needed, or just continue
            # For class mismatch, Ultralytics usually handles it by replacing the head when .train() is called with new data
            if model_type == 'RTDETR':
                model = RTDETR(weights)
            else:
                model = YOLO(weights)

            # Train
            # Using 20 epochs as "multiple rounds"
            train_name = name + '_train'
            model.train(data=DATA_YAML, epochs=20, imgsz=640, project=project_dir, name=train_name, batch=4)
            
            # After training, the best model is in project_dir/name_train/weights/best.pt
            best_pt = os.path.join(project_dir, train_name, 'weights', 'best.pt')
            print(f"\nTraining finished. Validating best model: {best_pt}")
            
            if os.path.exists(best_pt):
                if model_type == 'RTDETR':
                    model = RTDETR(best_pt)
                else:
                    model = YOLO(best_pt)
                    
                val_name = name + '_best_val'
                model.val(data=DATA_YAML, project=project_dir, name=val_name, split='test')
                
                # Update source for plots to the training/val run
                train_run_dir = os.path.join(project_dir, train_name) # Training plots
                val_run_dir = os.path.join(project_dir, val_name) # Val plots
                
                print("Copying plots to output directory...")
                # Copy plots from training run (results.png shows loss/metrics over epochs)
                for plot in ['results.png', 'confusion_matrix.png', 'F1_curve.png', 'PR_curve.png', 'confusion_matrix_normalized.png']:
                    safe_copy(os.path.join(train_run_dir, plot), os.path.join(out_dir, "train_" + plot))
                
                # Copy plots from val run
                for plot in ['confusion_matrix.png', 'F1_curve.png', 'PR_curve.png', 'confusion_matrix_normalized.png']:
                     safe_copy(os.path.join(val_run_dir, plot), os.path.join(out_dir, plot))
            else:
                print("Training finished but best.pt not found.")

        except Exception as e2:
            print(f"Training failed: {e2}")

if __name__ == '__main__':
    # Process RTDETR
    # process_model('RTDETR', RTDETR_WEIGHTS, RTDETR_DIR, 'rtdetr_eval')
    
    # Process YOLO
    process_model('YOLO', YOLO_WEIGHTS, YOLO_DIR, 'yolo_eval')
    
    # Run RTDETR separately or sequentially? Sequentially is safer for memory.
    process_model('RTDETR', RTDETR_WEIGHTS, RTDETR_DIR, 'rtdetr_eval')
