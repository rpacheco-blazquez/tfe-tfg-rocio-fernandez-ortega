# Blender

This folder contains Blender scripts and files for importing SeaFEM simulation results and generating visualization frames.

## Purpose

Import mesh and wave elevation results from SeaFEM simulations into Blender, set up scenes, and export frames for the training dataset.

## Workflow

1. Import mesh data from SeaFEM
2. Import wave elevation results
3. Set up camera angles and lighting
4. Configure animation timeline
5. Render and export frames as .png or .jpg files

## File Organization

- `scripts/` - Python scripts for Blender automation
- `scenes/` - Blender scene files (.blend)
- `exports/` - Rendered frames ready for Roboflow

## Requirements

- Blender 3.x or higher
- Python Blender API (bpy)
- Sufficient disk space for rendered frames

## Usage

Run Blender scripts in batch mode:
```bash
blender --background --python import_seafem.py
```
