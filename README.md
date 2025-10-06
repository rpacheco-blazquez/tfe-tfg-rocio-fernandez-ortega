# tfe-tfg-rocio-fernandez-ortega

## Computer Vision Project using YOLO

This project implements a computer vision pipeline using YOLO from Ultralytics for wave detection and analysis.

## Pipeline Overview

The project follows a 5-stage pipeline:

1. **Simulation** (`simulation/`) - CFD simulation using SeaFEM from TDYN suite
2. **Blender** (`blender/`) - Export mesh and wave elevation results from SeaFEM to Blender
3. **Roboflow** (`roboflow/`) - Generate frames (.png or .jpg) from Blender and prepare for upload to Roboflow repository
4. **Train** (`train/`) - Import Roboflow dataset and train YOLO models
5. **Postprocess** (`postprocess/`) - Process and analyze YOLO detections

## Project Structure

```
.
├── simulation/      # SeaFEM CFD simulation files
├── blender/         # Blender export scripts and files
├── roboflow/        # Frame generation and Roboflow integration
├── train/           # YOLO training scripts and models
├── postprocess/     # Detection postprocessing scripts
└── README.md        # This file
```

## Requirements

- Python 3.8+
- Ultralytics YOLO
- Blender (for visualization and frame generation)
- SeaFEM from TDYN suite (for CFD simulation)
- Roboflow account (for dataset management)

## Getting Started

### Quick Start
For a rapid introduction, see [QUICKSTART.md](QUICKSTART.md)

### Detailed Workflow
For comprehensive step-by-step instructions, see [WORKFLOW.md](WORKFLOW.md)

### Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Follow the pipeline stages in order, starting with `simulation/`

Each folder contains its own README with specific instructions.

## Documentation

- [QUICKSTART.md](QUICKSTART.md) - Get started quickly
- [WORKFLOW.md](WORKFLOW.md) - Complete pipeline guide
- `simulation/README.md` - CFD simulation setup
- `blender/README.md` - Frame generation with Blender
- `roboflow/README.md` - Dataset management with Roboflow
- `train/README.md` - YOLO model training
- `postprocess/README.md` - Detection postprocessing

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.