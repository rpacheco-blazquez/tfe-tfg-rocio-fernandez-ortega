"""
Postprocess YOLO detection results.

This script processes raw YOLO detection outputs, applies filters,
and generates statistics and visualizations.

Usage:
    python postprocess_detections.py --input detections/ --output results/ --confidence 0.5
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict


def load_detections(detections_dir):
    """
    Load detection results from directory.
    
    Args:
        detections_dir: Directory containing detection files
    
    Returns:
        List of detection dictionaries
    """
    detections = []
    detection_files = Path(detections_dir).glob("*.json")
    
    for file_path in detection_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            detections.append(data)
    
    return detections


def filter_by_confidence(detections, threshold=0.5):
    """
    Filter detections by confidence threshold.
    
    Args:
        detections: List of detection dictionaries
        threshold: Minimum confidence threshold
    
    Returns:
        Filtered detections
    """
    filtered = []
    
    for detection in detections:
        if 'confidence' in detection and detection['confidence'] >= threshold:
            filtered.append(detection)
    
    return filtered


def compute_statistics(detections):
    """
    Compute statistics from detections.
    
    Args:
        detections: List of detection dictionaries
    
    Returns:
        Dictionary containing statistics
    """
    if not detections:
        return {}
    
    confidences = [d['confidence'] for d in detections if 'confidence' in d]
    classes = [d['class'] for d in detections if 'class' in d]
    
    # Count detections per class
    class_counts = defaultdict(int)
    for cls in classes:
        class_counts[cls] += 1
    
    stats = {
        'total_detections': len(detections),
        'mean_confidence': np.mean(confidences) if confidences else 0,
        'std_confidence': np.std(confidences) if confidences else 0,
        'min_confidence': np.min(confidences) if confidences else 0,
        'max_confidence': np.max(confidences) if confidences else 0,
        'class_counts': dict(class_counts)
    }
    
    return stats


def visualize_statistics(stats, output_dir):
    """
    Create visualizations from statistics.
    
    Args:
        stats: Statistics dictionary
        output_dir: Directory to save visualizations
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Plot class distribution
    if 'class_counts' in stats:
        plt.figure(figsize=(10, 6))
        classes = list(stats['class_counts'].keys())
        counts = list(stats['class_counts'].values())
        plt.bar(classes, counts)
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Detection Count by Class')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path / 'class_distribution.png')
        plt.close()
        print(f"Saved class distribution plot to {output_path / 'class_distribution.png'}")


def save_results(detections, stats, output_dir):
    """
    Save processed results to files.
    
    Args:
        detections: Processed detections
        stats: Statistics dictionary
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save detections as JSON
    with open(output_path / 'processed_detections.json', 'w') as f:
        json.dump(detections, f, indent=2)
    
    # Save statistics as JSON
    with open(output_path / 'statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Save statistics as readable text
    with open(output_path / 'statistics.txt', 'w') as f:
        f.write("Detection Statistics\n")
        f.write("=" * 50 + "\n\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Results saved to {output_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Postprocess YOLO detection results"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Directory containing detection results"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/",
        help="Output directory for processed results"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not Path(args.input).exists():
        print(f"Error: Input directory not found: {args.input}")
        return
    
    print(f"Loading detections from {args.input}...")
    detections = load_detections(args.input)
    print(f"Loaded {len(detections)} detections")
    
    # Filter by confidence
    print(f"Filtering by confidence threshold: {args.confidence}")
    filtered_detections = filter_by_confidence(detections, args.confidence)
    print(f"Remaining detections: {len(filtered_detections)}")
    
    # Compute statistics
    print("Computing statistics...")
    stats = compute_statistics(filtered_detections)
    
    # Print statistics
    print("\nStatistics:")
    print("-" * 50)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Save results
    save_results(filtered_detections, stats, args.output)
    
    # Generate visualizations if requested
    if args.visualize:
        print("\nGenerating visualizations...")
        visualize_statistics(stats, args.output)
    
    print("\nPostprocessing completed successfully!")


if __name__ == "__main__":
    main()
