# Project Workflow Guide

This document provides a step-by-step guide to the complete pipeline for the wave detection computer vision project.

## Prerequisites

1. Install Python 3.8 or higher
2. Install Blender 3.x or higher
3. Install SeaFEM from TDYN suite
4. Create a Roboflow account
5. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Pipeline Stages

### Stage 1: Simulation (SeaFEM CFD)

**Location:** `simulation/`

**Objective:** Perform CFD simulations to generate wave data

**Steps:**
1. Set up SeaFEM simulation parameters
2. Define mesh and boundary conditions
3. Run CFD simulation
4. Export results:
   - Mesh file (e.g., `mesh.obj`)
   - Wave elevation data (e.g., `wave_elevation.csv`)
   - Metadata (e.g., `metadata.json`)

**Documentation:** See `simulation/README.md` and `simulation/OUTPUT_TEMPLATE.md`

### Stage 2: Blender Visualization

**Location:** `blender/`

**Objective:** Import simulation results and generate training frames

**Steps:**
1. Import mesh from SeaFEM:
   ```bash
   blender --background --python scripts/import_seafem.py
   ```
2. Import wave elevation data
3. Set up scene (camera, lighting, materials)
4. Configure animation timeline
5. Render frames to PNG/JPG
6. Export frames to `roboflow/frames/`

**Documentation:** See `blender/README.md`

**Script:** `blender/scripts/import_seafem.py`

### Stage 3: Roboflow Upload

**Location:** `roboflow/`

**Objective:** Upload frames to Roboflow for annotation

**Steps:**
1. Organize rendered frames in `roboflow/frames/`
2. Set Roboflow API key:
   ```bash
   export ROBOFLOW_API_KEY="your_api_key"
   ```
3. Upload frames:
   ```bash
   python upload_scripts/upload_to_roboflow.py \
     --api-key $ROBOFLOW_API_KEY \
     --workspace YOUR_WORKSPACE \
     --project YOUR_PROJECT \
     --images-dir frames/
   ```
4. Annotate images in Roboflow web interface
5. Generate dataset version
6. Export dataset in YOLO format

**Documentation:** See `roboflow/README.md`

**Script:** `roboflow/upload_scripts/upload_to_roboflow.py`

### Stage 4: YOLO Training

**Location:** `train/`

**Objective:** Train YOLO model on annotated dataset

**Steps:**
1. Download dataset from Roboflow (optional):
   ```bash
   python scripts/train_yolo.py \
     --download-dataset \
     --roboflow-api-key $ROBOFLOW_API_KEY \
     --workspace YOUR_WORKSPACE \
     --project YOUR_PROJECT \
     --version 1 \
     --data datasets/data.yaml \
     --model yolov8n.pt \
     --epochs 100
   ```

2. Or train with existing dataset:
   ```bash
   python scripts/train_yolo.py \
     --data datasets/data.yaml \
     --model yolov8n.pt \
     --epochs 100 \
     --img-size 640 \
     --batch-size 16
   ```

3. Validate model:
   ```bash
   python scripts/validate.py \
     --model models/yolo_wave_detection/weights/best.pt \
     --data datasets/data.yaml
   ```

**Documentation:** See `train/README.md`

**Scripts:** 
- `train/scripts/train_yolo.py`
- `train/scripts/validate.py`

### Stage 5: Postprocessing

**Location:** `postprocess/`

**Objective:** Process and analyze detection results

**Steps:**
1. Run inference with trained model (using Ultralytics CLI or API)
2. Process detection results:
   ```bash
   python scripts/postprocess_detections.py \
     --input detections/ \
     --output results/ \
     --confidence 0.5 \
     --visualize
   ```
3. Review generated statistics and visualizations in `results/`
4. Generate final reports

**Documentation:** See `postprocess/README.md`

**Script:** `postprocess/scripts/postprocess_detections.py`

## Model Selection Guide

Choose YOLO model variant based on requirements:

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| YOLOv8n | Fastest | Low | Real-time applications, edge devices |
| YOLOv8s | Fast | Medium | Balanced performance |
| YOLOv8m | Medium | Medium-High | Good balance |
| YOLOv8l | Slow | High | High accuracy needed |
| YOLOv8x | Slowest | Highest | Maximum accuracy |

## Common Issues and Solutions

### Issue: Blender script fails to import
**Solution:** Ensure Blender is run with `--background` flag and correct Python path

### Issue: Low model accuracy
**Solutions:**
- Increase training epochs
- Use larger model variant
- Add more training data
- Verify annotations quality
- Adjust augmentation parameters

### Issue: GPU not detected
**Solution:** Install CUDA-compatible PyTorch:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Tips for Best Results

1. **Data Quality:** Ensure consistent lighting and camera angles in Blender renders
2. **Annotations:** Double-check annotations in Roboflow for accuracy
3. **Training:** Start with a pre-trained model (transfer learning)
4. **Validation:** Use separate validation set to prevent overfitting
5. **Augmentation:** Use Roboflow's augmentation features to increase dataset diversity
6. **Monitoring:** Track training metrics (loss, mAP, precision, recall)

## Next Steps

After completing the pipeline:
1. Deploy model for inference
2. Optimize model for production (quantization, pruning)
3. Set up continuous training pipeline
4. Monitor model performance in production
5. Collect feedback and iterate

## Support

For issues with:
- **SeaFEM:** Refer to TDYN documentation
- **Blender:** Check Blender documentation and forums
- **Roboflow:** Contact Roboflow support
- **Ultralytics YOLO:** See [Ultralytics Docs](https://docs.ultralytics.com/)

## References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Roboflow Documentation](https://docs.roboflow.com/)
- [Blender Python API](https://docs.blender.org/api/current/)
- [SeaFEM Documentation](https://www.compassis.com/seafem)
