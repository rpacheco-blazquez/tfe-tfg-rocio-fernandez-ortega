# Postprocess

This folder contains scripts for postprocessing YOLO detection results and generating final outputs.

## Purpose

Process raw YOLO detection outputs, apply filters, aggregate results, and generate visualizations and reports.

## Workflow

1. Load YOLO detection results
2. Apply postprocessing filters (confidence thresholds, NMS, etc.)
3. Track objects across frames (if applicable)
4. Generate statistics and metrics
5. Create visualizations and reports

## File Organization

- `scripts/` - Postprocessing scripts
- `detections/` - Raw YOLO detection outputs
- `results/` - Processed results and visualizations
- `reports/` - Generated analysis reports

## Postprocessing Tasks

- Confidence filtering
- Non-maximum suppression (NMS)
- Object tracking across frames
- Statistical analysis
- Visualization generation
- Report creation

## Requirements

- NumPy
- Pandas
- Matplotlib/Seaborn
- OpenCV (for video processing)
- Ultralytics (for detection loading)

## Usage

Process detections:
```bash
python postprocess_detections.py --input detections/ --output results/ --confidence 0.5
```

Generate report:
```bash
python generate_report.py --detections results/processed.json --output reports/
```

## Metrics

Common metrics calculated:
- Detection counts per frame
- Average confidence scores
- Precision and recall
- Confusion matrices
- Wave statistics (height, frequency, etc.)
