"""
Train YOLO model using Roboflow dataset.

This script downloads a dataset from Roboflow and trains a YOLO model.

Usage:
    python train_yolo.py --data datasets/data.yaml --model yolov8n.pt --epochs 100
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def download_dataset(api_key, workspace, project, version):
    """
    Download dataset from Roboflow.
    
    Args:
        api_key: Roboflow API key
        workspace: Roboflow workspace name
        project: Roboflow project name
        version: Dataset version number
    
    Returns:
        Path to the downloaded dataset
    """
    from roboflow import Roboflow
    
    rf = Roboflow(api_key=api_key)
    project_obj = rf.workspace(workspace).project(project)
    dataset = project_obj.version(version).download("yolov8")
    
    return dataset.location


def train_model(data_yaml, model_name, epochs, img_size, batch_size, device):
    """
    Train YOLO model.
    
    Args:
        data_yaml: Path to data.yaml file
        model_name: YOLO model name (e.g., 'yolov8n.pt')
        epochs: Number of training epochs
        img_size: Image size for training
        batch_size: Batch size
        device: Device to use ('cuda' or 'cpu')
    """
    # Load YOLO model
    model = YOLO(model_name)
    
    print(f"Training {model_name} for {epochs} epochs...")
    print(f"Dataset: {data_yaml}")
    print(f"Image size: {img_size}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    
    # Train model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        project="models",
        name="yolo_wave_detection",
        save=True,
        save_period=10,
        verbose=True
    )
    
    print("Training completed!")
    print(f"Results saved to: {results.save_dir}")
    
    return results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Train YOLO model with Roboflow dataset"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data.yaml file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model to use (yolov8n.pt, yolov8s.pt, etc.)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Image size for training"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device to use (0 for GPU, cpu for CPU)"
    )
    parser.add_argument(
        "--download-dataset",
        action="store_true",
        help="Download dataset from Roboflow before training"
    )
    parser.add_argument(
        "--roboflow-api-key",
        type=str,
        default=None,
        help="Roboflow API key (required if --download-dataset is used)"
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Roboflow workspace (required if --download-dataset is used)"
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Roboflow project (required if --download-dataset is used)"
    )
    parser.add_argument(
        "--version",
        type=int,
        default=1,
        help="Dataset version (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Download dataset if requested
    if args.download_dataset:
        if not all([args.roboflow_api_key, args.workspace, args.project]):
            print("Error: --roboflow-api-key, --workspace, and --project required when using --download-dataset")
            return
        
        print("Downloading dataset from Roboflow...")
        dataset_path = download_dataset(
            api_key=args.roboflow_api_key,
            workspace=args.workspace,
            project=args.project,
            version=args.version
        )
        print(f"Dataset downloaded to: {dataset_path}")
    
    # Validate data.yaml exists
    if not Path(args.data).exists():
        print(f"Error: data.yaml not found at {args.data}")
        return
    
    # Train model
    train_model(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch_size,
        device=args.device
    )


if __name__ == "__main__":
    main()
