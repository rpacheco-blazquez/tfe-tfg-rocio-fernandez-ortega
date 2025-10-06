"""
Validate trained YOLO model.

This script validates a trained YOLO model on a test dataset.

Usage:
    python validate.py --model models/best.pt --data datasets/data.yaml
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def validate_model(model_path, data_yaml, img_size, batch_size, device):
    """
    Validate YOLO model on test dataset.
    
    Args:
        model_path: Path to trained model weights
        data_yaml: Path to data.yaml file
        img_size: Image size for validation
        batch_size: Batch size
        device: Device to use ('cuda' or 'cpu')
    
    Returns:
        Validation results
    """
    # Load model
    model = YOLO(model_path)
    
    print(f"Validating model: {model_path}")
    print(f"Dataset: {data_yaml}")
    print(f"Image size: {img_size}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    
    # Validate
    results = model.val(
        data=data_yaml,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        save_json=True,
        save_hybrid=True,
        conf=0.001,
        iou=0.6,
        max_det=300,
        plots=True
    )
    
    # Print results
    print("\nValidation Results:")
    print("-" * 50)
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall: {results.box.mr:.4f}")
    
    return results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Validate YOLO model"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model weights"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data.yaml file"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Image size for validation"
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
    
    args = parser.parse_args()
    
    # Validate model exists
    if not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}")
        return
    
    # Validate data.yaml exists
    if not Path(args.data).exists():
        print(f"Error: data.yaml not found at {args.data}")
        return
    
    # Validate model
    validate_model(
        model_path=args.model,
        data_yaml=args.data,
        img_size=args.img_size,
        batch_size=args.batch_size,
        device=args.device
    )
    
    print("\nValidation completed successfully!")


if __name__ == "__main__":
    main()
