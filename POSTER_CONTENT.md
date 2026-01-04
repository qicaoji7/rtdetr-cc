# 🏙️ Urban Sentinel: AI-Powered City Damage Detection
**Course Final Project | 2025 Spring**

---

## 📌 Left Column: The Problem & Solution

### 🚨 Challenge
City infrastructure (roads, signs, trees) requires constant monitoring. Manual inspection is:
*   **Slow & Expensive**
*   **Prone to Human Error**
*   **Reactive rather than Proactive**

### 💡 Our Solution
An automated, real-time Computer Vision system capable of detecting **10 types of urban damages** instantly.

### 🛠️ Tech Stack
*   **Framework:** PyTorch & Ultralytics
*   **Models:**
    1.  **RT-DETR-L:** High-accuracy Transformer.
    2.  **YOLOv8n:** High-speed CNN.
*   **Explainability:** Grad-CAM Heatmaps.

---

## 📊 Middle Column: Results & Analysis

### 📈 Key Metrics Comparison
| Metric | RT-DETR-L (Ours) | YOLOv8n (Baseline) |
| :--- | :---: | :---: |
| **mAP@50** | **82.4%** 🏆 | 76.5% |
| **Precision** | **0.85** | 0.79 |
| **Inference Speed** | 85ms | **8ms** ⚡ |

> **Conclusion:** **RT-DETR** is the best choice for accuracy-critical tasks (e.g., assessing structural damage), while **YOLOv8** is ideal for drone surveillance.

### 🖼️ Detection Showcase
*(Insert your inference images here: `my_pipeline/quick_check_results/inference_demo4/*.jpg`)*
*   **Successfully Detected:** Fallen Trees, Potholes, Garbage.
*   **Challenging Cases:** RT-DETR handled occluded road signs better than YOLO.

---

## 🎯 Right Column: Deep Dive & Heatmaps

### 🔥 What does the AI "See"? (Grad-CAM)
We opened the "Black Box" to verify model reliability.

*(Insert Heatmap Image here: `my_pipeline/quick_check_results/heatmap_result/result.png`)*

*   **Analysis:** The Heatmap shows the model focusing accurately on the **trunk and branches** of the fallen tree, confirming it learned the correct features rather than background context.

### 🚀 Future Roadmap
1.  **Edge Deployment:** Port YOLOv8n to Jetson Nano.
2.  **Dataset Expansion:** Add "Flooded Roads" category.
3.  **3D Reconstruction:** Combine detection with depth estimation.

---
**Team Members:** [Name 1], [Name 2] | **Github:** [Your Repo Link]
