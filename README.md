# ML-DETR: Multiscale-Lite Detection Transformer for Mature Cherry Tomato Identification

[![Paper](https://img.shields.io/badge/Paper-IEEE%20TIM%202025-blue)](https://doi.org/10.1109/TIM.2025.3608317)
![Q1](https://img.shields.io/badge/JCR-Q1-green)
![CCF-B](https://img.shields.io/badge/CCF-B-orange)
![IF](https://img.shields.io/badge/IF-7.2-red)

Official implementation of the paper:

> **ML-DETR: Multiscale-Lite Detection Transformer for Identification of Mature Cherry Tomatoes**
> Zihao Tang, Liangfei Fang, Shoujun Sun, et al.
> *IEEE Transactions on Instrumentation & Measurement*, Vol. 74, 2025.
> JCR Q1 · CAS Zone 2 (TOP) · CCF-B · IF 7.2
> DOI: [10.1109/TIM.2025.3608317](https://doi.org/10.1109/TIM.2025.3608317)

## Highlights

- **SAMS-CAM**: a Self-Attention Multi-Scale Collection and Allocation Module that explicitly allocates attention across feature scales, targeting small, occluded fruits in cluttered greenhouse scenes.
- **Lite design for edge AI**: a lightweight DETR-style detector that reaches **98.7% recognition accuracy on the NVIDIA Jetson TX1** embedded platform.
- **Field-validated**: deployed on an air-ground collaborative harvesting robot; 100+ km of field debugging; harvesting efficiency **+35%**, energy consumption **−43%**, zero carbon emissions.
- Full pipeline included: training, validation, detection, tracking (ByteTrack/BoT-SORT), FPS benchmarking, ERF visualization, and heatmap analysis.

## Model Architecture

```
Input image
   └─ Lightweight CNN backbone (replaceable: MobileNet / Ghost / ShuffleNet / EfficientFormer / ...)
        └─ Multi-scale feature collection
             └─ SAMS-CAM (self-attention multi-scale allocation)  ← key contribution
                  └─ Transformer decoder (object queries)
                       └─ Detection head (class + box)
```

## Main Results

| Model | Backbone | mAP@50 (%) | Params (M) | FPS (Jetson TX1) |
|---|---|---|---|---|
| YOLOv5 | CSPDarkNet | 【以论文表格为准】 | 【】 | 【】 |
| YOLOv8 | — | 【】 | 【】 | 【】 |
| RT-DETR | ResNet-50 | 【】 | 【】 | 【】 |
| **ML-DETR (ours)** | Lite backbone | **【】** | 【】 | **98.7% accuracy · 【FPS】** |

> Note: exact comparison numbers are filled from the published paper tables (Section: Comparison with SOTA / Ablation Study).

## Dataset

A self-built multi-view cherry tomato dataset covering three cultivars (purple / red / yellow cherry tomatoes), captured at multiple distances, angles and orientations in real greenhouses, including occlusion and complex backgrounds.

- Annotation: [labelImg](https://github.com/HumanSignal/labelImg), YOLO/COCO formats
- Scripts: `dataset/split_data.py`, `dataset/xml2txt.py`, `dataset/yolo2coco.py`, `dataset/data2plus.py`
- The dataset is currently **private** due to an ongoing commercialization project; it can be shared for academic review on request.

## Installation

```bash
# Python 3.9 + CUDA 11.6 (the environment this project was developed in)
git clone https://github.com/【你的用户名】/ML-DETR.git
cd ML-DETR
pip install -r requirements.txt
pip install -e .
```

## Usage

```bash
# Train
python train.py --data dataset/data.yaml --model rtdetr-mamba-T --imgsz 640 --epochs 300

# Validate
python val.py --data dataset/data.yaml --weights runs/train/exp/best.pt

# Detect images / videos
python detect.py --weights best.pt --source path/to/video_or_image

# Fruit tracking in videos (ByteTrack / BoT-SORT)
python track.py --weights best.pt --source greenhouse.mp4

# Edge-speed evaluation & ERF/heatmap analysis
python get_FPS.py
python get_model_erf.py
python heatmap.py
```

Model configs live in `ultralytics/cfg/models/rt-detr/` (RT-DETR series, Mamba variants) and `ultralytics/nn/backbone/` (FasterNet, MobileNetV4, RepViT, EfficientFormerV2, Swin / CSwin Transformer, Mamba, etc.).

## Experiments Included

1. Comparison with SOTA detectors (YOLOv5 / v8, RT-DETR, Faster R-CNN)
2. Data augmentation comparison
3. Ablation study of SAMS-CAM and key modules
4. Knowledge distillation experiments (`RTDETR-KG/RTDETR-distill`)
5. Lightweight backbone replacement studies
6. Accuracy / parameter count / FPS trade-off analysis

## Deployment

- Exported to ONNX and deployed on **NVIDIA Jetson TX1**;
- detector + ByteTrack used for fruit tracking to guide an 8-DOF bionic arm;
- EKF fusion of LiDAR + IMU + wheel odometry provides field navigation within **3 cm** error.

## Citation

```bibtex
@article{tang2025mldetr,
  title={ML-DETR: Multiscale-Lite Detection Transformer for Identification of Mature Cherry Tomatoes},
  author={Tang, Zihao and Fang, Liangfei and Sun, Shoujun and others},
  journal={IEEE Transactions on Instrumentation and Measurement},
  volume={74},
  year={2025},
  doi={10.1109/TIM.2025.3608317}
}
```

## Acknowledgement

Built on top of [ultralytics](https://github.com/ultralytics/ultralytics) and RT-DETR. Thanks to the team and the greenhouse bases in Chaohu, Anhui for field-test support.

---

### 中文简介

本仓库是我以**唯一第一作者**发表于 IEEE TIM（JCR Q1 / 中科院二区TOP / CCF-B / IF 7.2）的论文配套代码。针对设施农业中成熟樱桃番茄的小目标、遮挡与复杂背景问题，提出轻量检测网络 **ML-DETR** 与核心模块 **SAMS-CAM**，并在 Jetson TX1 嵌入式平台实现 98.7% 的识别准确率，已在真实采收机器人上完成田间部署验证。
