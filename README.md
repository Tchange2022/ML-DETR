# ML-DETR: Multiscale-Lite Detection Transformer for Mature Cherry Tomato Identification

[![Paper](https://img.shields.io/badge/Paper-IEEE%20TIM%202025-blue)](https://doi.org/10.1109/TIM.2025.3608317)
![Q1](https://img.shields.io/badge/JCR-Q1-green)
![CCF-B](https://img.shields.io/badge/CCF--B-orange)
![IF](https://img.shields.io/badge/IF-7.2-red)

Official implementation of the paper:

> **ML-DETR: Multiscale-Lite Detection Transformer for Identification of Mature Cherry Tomatoes**
> Zihao Tang, Liangfei Fang, Shoujun Sun, Yue Gong, and Qing Li
> *IEEE Transactions on Instrumentation & Measurement*, Vol. 74, 2025 (Article no. 2547018)
> JCR Q1 · CAS Zone 2 (TOP) · CCF-B · IF 7.2
> DOI: [10.1109/TIM.2025.3608317](https://doi.org/10.1109/TIM.2025.3608317)

## Highlights

- **SAMS-CAM**: a self-attention multi-scale collection-and-allocation module. In code it is realized by a **3D multi-scale collection block (`ScalSeq`)** that stacks P3/P4/P5 features and fuses them along the scale axis with a Conv3D, **cross-level feature injection (`SimFusion` / `InjectionMultiSum`)**, and the **self-/cross-attention layers inside the `RTDETRDecoder`** — targeting small, occluded fruits across cluttered greenhouse scenes (paper Section II, Fig. 2).
- **Lightweight for edge AI**: **13.9 GFLOPs** final model (Table II, Order 11) — a **76.2% compute reduction vs. RT-DETR-R18** (58.3 GFLOPs). The repo profiler reports 14.2 GFLOPs / 9.06 M params with the default 80-class COCO head; the paper uses the single-class `rc_tomato` head (13.9 GFLOPs).
- **Published results**: **91.36% precision, 0.8875 mAP50, 0.5774 mAP50-95, 85.50% F1** on the augmented test set (Tables I–III), **81.2 FPS** on an RTX 3050 Ti laptop GPU.
- **Jetson TX1 deployment (paper Section IV-B)**: **72.3 FPS** inference, **13.8 ms** end-to-end latency, **23.6 W** average power, 4.2 GB peak memory, <1% thermal throttling.
- Full pipeline: training, validation, image/video detection, tracking, FPS benchmarking, ERF visualization and heat-map analysis; plus knowledge-distillation experiments (`RTDETR-KG/`).

## Model Architecture

Final config: [`ultralytics/cfg/models/yolo-detr/yolov8-detr-goldyolo-asf-RGCSPELAN.yaml`](ultralytics/cfg/models/yolo-detr/yolov8-detr-goldyolo-asf-RGCSPELAN.yaml) (508 layers).

```
Input (640x640)
  └─ YOLOv8n-style lightweight CNN backbone, basic block = RGCSPELAN (Rep Ghost CSP-ELAN, RepConv re-parameterization)
       └─ P3/P4/P5 multi-scale features
            └─ ScalSeq            -> 3D cross-scale collection (Conv3D along the scale axis)  ] SAMS-CAM
               SimFusion_3in/4in  -> multi-scale feature gathering                            ] (collection
               InjectionMultiSum   -> cross-level feature allocation/injection               ]  & allocation)
            └─ RTDETRDecoder       -> object-query self-attention + cross-attention + detection head (class + box)
```

Paper-module ↔ code mapping (the paper uses conceptual names; the implementation reuses engineering class names):

| Paper concept | Code |
|---|---|
| Multi-scale collection (SSFF / scale-space) | `ScalSeq` (`ultralytics/nn/extra_modules/block.py:2355`) |
| Multi-scale allocation / cross-stage fusion | `SimFusion_4in/3in`, `InjectionMultiSum_Auto_pool`, `AdvPoolFusion` (`block.py`) |
| Transformer self-/cross-attention fusion | `RTDETRDecoder` (`ultralytics/nn/modules/head.py`) |
| Re-parameterization (REP) | `RepConv` inside `RGCSPELAN` (`block.py:4757`) |
| Cross-stage feature enhancement (SCFENet) | stacked `RGCSPELAN` stages with the injection neck |

## Main Results (paper Table III — comparison with 23 SOTA methods)

| Method | GFLOPs | FPS* | mAP50 | mAP50-95 | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| RT-DETR-R18 | 58.3 | 85.4 | 0.73883 | 0.37734 | 65.27% | 77.28% | 70.76% |
| PP-DETR | 44.9 | 68 | 0.802 | 0.423 | 80.40% | 78.20% | 79.30% |
| YOLOv10s | 21.6 | 90 | 0.854 | 0.521 | 84.70% | 80.10% | 82.30% |
| YOLOv9-S | 21.6 | 95 | 0.868 | 0.532 | 85.20% | 81.30% | 83.20% |
| GOLD-YOLO | 28.3 | 105 | 0.879 | 0.558 | 87.10% | 82.60% | 84.80% |
| YOLOv8n | 8.9 | 75.1 | 0.87631 | 0.56641 | 88.08% | 78.38% | 82.95% |
| YOLOv5n | 16.0 | 32.7 | 0.65316 | 0.32795 | 61.88% | 72.69% | 66.85% |
| Faster R-CNN | 251.4 | 8.2 | 0.86497 | 0.57836 | 92.03% | 80.59% | 85.93% |
| **ML-DETR (ours)** | **13.9** | **81.2** | **0.88751** | **0.57738** | **91.36%** | **80.34%** | **85.50%** |

\* FPS measured on an NVIDIA GeForce RTX 3050 Ti (4 GB) laptop, Intel i5-12500H, Windows 11, 640×640, batch 8 (paper Section IV-B). Jetson TX1 speed: **72.3 FPS**.

## Ablation Study (paper Table II, selected orders; augmented dataset, 640×640, batch 8)

| Order | Backbone / modules (✓) | GFLOPs | FPS | Precision | Recall | mAP50 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | RT-DETR-R18 | 58.3 | 85.4 | 65.27% | 77.28% | 0.7388 |
| 4 | YOLOv8n | 8.9 | 75.1 | 88.08% | 78.38% | 0.8663 |
| 5 | R18 + YOLOv8n | 11.8 | 76.9 | 83.63% | 78.45% | 0.8597 |
| 6 | + GOLD-YOLO neck (fusion baseline) | 12.1 | 43.2 | 82.01% | 77.99% | 0.8291 |
| 7 | + SAMS-CAM | 12.2 | 75.7 | 86.93% | 80.70% | 0.8364 |
| 8 | + SCFENet | 14.3 | 72.5 | 87.64% | 79.64% | 0.8370 |
| 9 | + REP | 9.8 | 79.6 | 82.65% | 81.37% | 0.7974 |
| 10 | + SAMS-CAM + SCFENet (no REP) | 14.5 | 74.1 | 89.45% | 73.60% | 0.8746 |
| **11** | **Full: + SAMS-CAM + SCFENet + REP (ML-DETR)** | **13.9** | **81.2** | **91.36%** | **80.34%** | **0.8875** |

Table II contains 17 orders in total (also covering R34/R50 backbones, YOLOv8-s/m/l variants, and 1024×1024 / batch-16 settings).

## Data Augmentation (paper Table I)

ML-DETR test-set mAP50 improved from **0.6659 (original 3,432-image set)** to **0.8875 (augmented 8,580-image set)**; precision from 77.70% to 91.36%.

## Generalizability (paper Table V — five public datasets)

| Grape (PlantVillage) | Apple (PlantVillage) | CUB-200-2011 | Stanford Cars | Stanford Dogs |
|---:|---:|---:|---:|---:|
| 90.62% | 90.16% | 83.41% | 81.75% | 80.67% |

ML-DETR outperformed Faster R-CNN by 3.2%–7.5% across these datasets while using only 5.5% of its compute (paper Section IV-F).

## Dataset

- **Single-class detection** (`nc: 1`, class name `rc_tomato` = ripe cherry tomato), see `dataset/data.yaml`.
- **3,432 original images**, captured with a VIVO X100 at the tomato base in Xiaolianwei village, Zhong'an town, Chaohu, Anhui on 2024-01-20.
- Seven acquisition conditions: camera angle, fruit overlap/occlusion, different backgrounds, variable lighting, darkening, partial occlusion, incomplete display.
- **8,580 images after augmentation**, randomly split **train : val : test = 8 : 1 : 1**; COCO-pretrained initialization, 640×640, 100 epochs, Adam optimizer.
- Annotation: [labelImg](https://github.com/HumanSignal/labelImg), YOLO/COCO/XML formats.
- Scripts: `dataset/split_data.py`, `dataset/xml2txt.py`, `dataset/txt2xml.py`, `dataset/yolo2coco.py`, `dataset/data2plus.py`, `dataset/ahchorsize.py`.
- The image data is **private** due to an ongoing commercialization project; it can be shared for academic review on request.

## Installation

```bash
git clone https://github.com/Tchange2022/ML-DETR.git
cd ML-DETR
pip install -r requirements.txt
pip install -e .
```

Developed on Python 3.9 + CUDA 11.6 (paper); the repository was also verified locally with Python 3.8 + torch 1.11 (CUDA build). The Mamba variants (`rtdetr-mamba-{T,B,L}.yaml`) additionally require the `mamba_ssm` CUDA extension; see `test_env.py` for the optional CUDA extensions (mamba_ssm, DCNv3/DCNv4, mmcv).

## Usage

The entry scripts use hard-coded paths — edit the paths inside each file, then run directly:

```bash
# Train (final ML-DETR config, dataset/data.yaml, 640px, 100 epochs)
python train.py

# Validate on the test split (uses your trained best.pt)
python val.py

# Detect images / a folder (default source: dataset/images/test)
python detect.py

# Track objects in a video (Ultralytics BoT-SORT; switch to ByteTrack via cfg/trackers/bytetrack.yaml)
python track.py

# Speed / parameter profiling, ERF and heat-map analysis
python get_FPS.py
python get_model_erf.py
python heatmap.py
python get_COCO_metrice.py
```

Minimal API example:

```python
from ultralytics import RTDETR

# build the paper model from config (nc is read from dataset/data.yaml when training)
model = RTDETR('ultralytics/cfg/models/yolo-detr/yolov8-detr-goldyolo-asf-RGCSPELAN.yaml')
model.train(data='dataset/data.yaml', imgsz=640, epochs=100, batch=8, device='0')

# or load trained weights
model = RTDETR('runs/train/exp/weights/best.pt')
model.val(data='dataset/data.yaml', split='test', imgsz=640)
model.predict(source='dataset/images/test', save=True)
```

## Code Structure

```
├── train.py / val.py / detect.py / track.py      # training & inference entry points
├── get_FPS.py / get_model_erf.py / heatmap.py    # profiling & visualization
├── get_COCO_metrice.py / plot_result.py / main_profile.py
├── dataset/                                      # data split / format-conversion scripts (images not included)
├── ultralytics/
│   ├── cfg/models/rt-detr/                       # rtdetr-r18/r34/r50/r101, rtdetr-mamba-T/B/L
│   ├── cfg/models/yolo-detr/                     # ML-DETR final config + baselines
│   ├── cfg/trackers/                             # bytetrack.yaml, botsort.yaml
│   └── nn/
│       ├── backbone/                             # FasterNet, MobileNetV4, RepViT, EfficientFormerV2,
│       │                                         #   Swin/CSwin, TransNeXt, Mamba, etc.
│       └── extra_modules/block.py                # ScalSeq, RGCSPELAN, SimFusion, InjectionMultiSum, ...
└── RTDETR-KG/RTDETR-distill/                     # knowledge-distillation experiments (RTDETR-distill)
```

## Deployment (paper Section IV-B)

The final network runs on a self-developed picking robot's **NVIDIA Jetson TX1** (Tegra X1, 4× Cortex-A57 @1.9 GHz, 256-core Maxwell GPU @998 MHz, 4 GB LPDDR4). Measured operating metrics: 72.3 FPS inference, 13.8 ms end-to-end latency, 23.6 W average power, 4.2 GB peak memory, <1% thermal throttling, 78.4% CPU / 91.2% GPU utilization.

## Citation

```bibtex
@article{tang2025mldetr,
  title={ML-DETR: Multiscale-Lite Detection Transformer for Identification of Mature Cherry Tomatoes},
  author={Tang, Zihao and Fang, Liangfei and Sun, Shoujun and Gong, Yue and Li, Qing},
  journal={IEEE Transactions on Instrumentation and Measurement},
  volume={74},
  pages={2547018},
  year={2025},
  doi={10.1109/TIM.2025.3608317}
}
```

## Acknowledgement

Built on top of [ultralytics](https://github.com/ultralytics/ultralytics) and RT-DETR; the multi-scale neck references Gold-YOLO's fusion design. Thanks to the greenhouse bases in Chaohu, Anhui for field-test support.

---

### 中文简介

本仓库是我以**唯一第一作者**发表于 IEEE TIM（JCR Q1 / 中科院二区TOP / CCF-B / IF 7.2，2025年9月底正式发表）的论文配套代码。针对设施农业中成熟樱桃番茄的小目标、遮挡与复杂背景问题，提出轻量检测网络 **ML-DETR**：以 YOLOv8n 轻量主干为基础，通过 `ScalSeq` 三维多尺度采集、跨层特征注入分配与 RT-DETR 解码器的自注意力共同实现论文中的 **SAMS-CAM**，配合重参数化模块（RepConv）。在自建番茄数据集（3,432 张原图、增强后 8,580 张、7 类复杂环境、单类成熟果实检测）上取得 **91.36% 精确率、0.8875 mAP50、85.50% F1**，算力仅 **13.9 GFLOPs**（较 RT-DETR-R18 降低 76.2%），RTX 3050 Ti 上 **81.2 FPS**，并在 **Jetson TX1** 嵌入式平台实测 **72.3 FPS、13.8 ms 延迟、23.6 W 功耗**。
