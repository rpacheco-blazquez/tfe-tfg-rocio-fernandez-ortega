# Roboflow

This folder contains scripts for preparing frames and uploading them to the Roboflow platform for dataset management and annotation.

## Purpose

Organize and upload rendered frames from Blender to a Roboflow repository for labeling, augmentation, and dataset version control.

## Workflow

1. Collect frames from Blender exports
2. Organize frames by categories/classes
3. Upload to Roboflow via API
4. Annotate images in Roboflow platform
5. Export dataset for YOLO training

## File Organization

- `frames/` - Rendered frames from Blender
- `upload_scripts/` - Scripts to upload images to Roboflow
- `config/` - Roboflow API configuration

## Requirements

- Roboflow account and API key
- Python Roboflow package
- Organized frame structure

## Usage

Upload frames to Roboflow:
```bash
python upload_to_roboflow.py --api-key YOUR_API_KEY --project YOUR_PROJECT
```

## Environment Variables

Set your Roboflow API key:
```bash
export ROBOFLOW_API_KEY="your_api_key_here"
```
