import os
import sys
try:
    import rarfile
except ImportError:
    print("Error: rarfile library not found. Please install it using 'pip install rarfile'")
    sys.exit(1)

def unpack_rar(rar_path, extract_to):
    print(f"Extracting {rar_path} to {extract_to}...")
    try:
        # Check if unrar is available in path, if not rarfile might fail unless configured
        # This script assumes 'unrar' binary is available or 'rarfile' can handle it.
        # If rarfile pure python implementation is not enough, we might need system unrar.
        # Let's try basic extraction first.
        with rarfile.RarFile(rar_path) as rf:
            rf.extractall(extract_to)
        print("Extraction successful.")
    except rarfile.RarCannotExec as e:
        print(f"Error: unrar executable not found. 'rarfile' needs unrar installed on system.")
        print(f"Details: {e}")
    except Exception as e:
        print(f"Error extracting {rar_path}: {e}")

if __name__ == "__main__":
    os.makedirs("analysis_temp", exist_ok=True)
    
    redetr_path = "redetr.rar"
    yolo_path = "yolo8n.rar"
    
    if os.path.exists(redetr_path):
        unpack_rar(redetr_path, "analysis_temp/redetr")
    else:
        print(f"{redetr_path} not found.")
        
    if os.path.exists(yolo_path):
        unpack_rar(yolo_path, "analysis_temp/yolo8n")
    else:
        print(f"{yolo_path} not found.")
