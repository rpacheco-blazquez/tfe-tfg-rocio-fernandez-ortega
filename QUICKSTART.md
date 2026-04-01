# Quick Start Guide

Get started with the wave detection YOLO pipeline in minutes.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rpacheco-blazquez/tfe-tfg-rocio-fernandez-ortega.git
   cd tfe-tfg-rocio-fernandez-ortega
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify installation:
   ```bash
   python -c "from ultralytics import YOLO; print('YOLO installed successfully!')"
   ```

## Quick Pipeline Overview

### 1. Prepare Simulation Data (simulation/)
- Run SeaFEM CFD simulation
- Export mesh and wave elevation data

### 2. Generate Training Frames (blender/)
```bash
blender --background --python blender/scripts/import_seafem.py
```

### 3. Upload to Roboflow (roboflow/)
```bash
export ROBOFLOW_API_KEY="your_key"
python roboflow/upload_scripts/upload_to_roboflow.py \
  --api-key $ROBOFLOW_API_KEY \
  --workspace YOUR_WORKSPACE \
  --project YOUR_PROJECT \
  --images-dir roboflow/frames/
```

### 4. Train YOLO Model (train/)
```bash
python train/scripts/train_yolo.py \
  --data train/datasets/data.yaml \
  --model yolov8n.pt \
  --epochs 50
```

### 5. Process Detections (postprocess/)
```bash
python postprocess/scripts/postprocess_detections.py \
  --input postprocess/detections/ \
  --output postprocess/results/ \
  --visualize
```

## Example: Training with Pre-downloaded Dataset

If you already have a dataset from Roboflow:

1. Place dataset in `train/datasets/`
2. Train model:
   ```bash
   python train/scripts/train_yolo.py \
     --data train/datasets/data.yaml \
     --model yolov8s.pt \
     --epochs 100 \
     --batch-size 16
   ```
3. Validate:
   ```bash
   python train/scripts/validate.py \
     --model train/models/yolo_wave_detection/weights/best.pt \
     --data train/datasets/data.yaml
   ```

## Next Steps

- Read the full [WORKFLOW.md](WORKFLOW.md) for detailed instructions
- Check individual folder README files for stage-specific documentation
- Review [Ultralytics documentation](https://docs.ultralytics.com/) for advanced features

## Troubleshooting

**Import Error:** Make sure all dependencies are installed
```bash
pip install -r requirements.txt --upgrade
```

**GPU Not Found:** Install CUDA-compatible PyTorch
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Need Help?** See the [WORKFLOW.md](WORKFLOW.md) for common issues and solutions.
