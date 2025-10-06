# Simulation Output Template

This template describes the expected output structure from SeaFEM simulations.

## Output Directory Structure

```
simulation/
├── input/          # Simulation input files
├── output/         # Simulation results
│   ├── mesh.obj    # Exported mesh file
│   ├── wave_elevation.csv  # Wave elevation data
│   └── metadata.json       # Simulation metadata
└── README.md
```

## Expected File Formats

### Mesh Export (mesh.obj)
- Standard OBJ format compatible with Blender
- Contains vertices, faces, and normals
- Scale and coordinate system should be documented

### Wave Elevation Data (wave_elevation.csv)
Expected columns:
- `time`: Simulation time step
- `x`: X coordinate
- `y`: Y coordinate
- `z`: Z coordinate (elevation)
- Additional columns as needed

### Metadata (metadata.json)
Example structure:
```json
{
  "simulation_name": "wave_simulation_001",
  "date": "2024-01-01",
  "duration": 10.0,
  "timestep": 0.01,
  "wave_parameters": {
    "height": 2.0,
    "period": 5.0,
    "direction": 0
  }
}
```

## Coordinate System

Document the coordinate system used:
- Origin location
- Axis orientation (e.g., Z-up or Y-up)
- Units (meters, etc.)

## Notes

- Ensure all output files use consistent coordinate systems
- Include sufficient metadata for reproducibility
- Consider file size and compression for large datasets
