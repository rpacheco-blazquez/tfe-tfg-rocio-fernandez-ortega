# Train

This folder contains scripts and configurations for training YOLO models using the Roboflow dataset.

## Purpose

Import datasets from Roboflow and train YOLO (Ultralytics) models for wave detection and classification.

## Workflow

1. Download dataset from Roboflow
2. Configure YOLO training parameters
3. Train YOLO model
4. Validate model performance
5. Export trained model weights

## File Organization

- `datasets/` - Downloaded datasets from Roboflow
- `configs/` - YOLO configuration files (data.yaml, model configs)
- `models/` - Trained model weights
- `scripts/` - Training and validation scripts
- `results/` - Training logs, metrics, and visualizations

## Requirements

- Ultralytics YOLO (YOLOv8 or later)
- PyTorch
- CUDA-enabled GPU (recommended)
- Roboflow Python package

## Usage

Train a YOLO model:
```bash
python train_yolo.py --data datasets/data.yaml --model yolov8n.pt --epochs 100
```

Validate model:
```bash
python validate.py --model models/best.pt --data datasets/data.yaml
```

## Model Selection

Choose appropriate YOLO model size based on requirements:
- YOLOv8n: Nano (fastest, least accurate)
- YOLOv8s: Small
- YOLOv8m: Medium
- YOLOv8l: Large
- YOLOv8x: Extra large (slowest, most accurate)
