"""
Upload frames to Roboflow repository.

This script uploads image frames to Roboflow for annotation and dataset management.

Usage:
    python upload_to_roboflow.py --api-key YOUR_API_KEY --project YOUR_PROJECT
"""

import os
import argparse
from pathlib import Path
from roboflow import Roboflow


def upload_images(api_key, workspace, project_name, images_dir, batch_name=None):
    """
    Upload images to Roboflow project.
    
    Args:
        api_key: Roboflow API key
        workspace: Roboflow workspace name
        project_name: Roboflow project name
        images_dir: Directory containing images to upload
        batch_name: Optional batch name for organizing uploads
    """
    # Initialize Roboflow
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    
    # Get list of image files
    image_extensions = ['.png', '.jpg', '.jpeg']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(images_dir).glob(f"*{ext}"))
    
    print(f"Found {len(image_files)} images to upload")
    
    # Upload images
    for idx, image_path in enumerate(image_files, 1):
        try:
            print(f"Uploading {idx}/{len(image_files)}: {image_path.name}")
            project.upload(
                image_path=str(image_path),
                batch_name=batch_name or "default",
                num_retry_uploads=3
            )
        except Exception as e:
            print(f"Error uploading {image_path.name}: {e}")
            continue
    
    print("Upload completed!")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Upload frames to Roboflow repository"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="Roboflow API key"
    )
    parser.add_argument(
        "--workspace",
        type=str,
        required=True,
        help="Roboflow workspace name"
    )
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="Roboflow project name"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="../roboflow/frames/",
        help="Directory containing images to upload"
    )
    parser.add_argument(
        "--batch-name",
        type=str,
        default=None,
        help="Optional batch name for organizing uploads"
    )
    
    args = parser.parse_args()
    
    # Validate images directory
    if not os.path.exists(args.images_dir):
        print(f"Error: Images directory not found: {args.images_dir}")
        return
    
    # Upload images
    upload_images(
        api_key=args.api_key,
        workspace=args.workspace,
        project_name=args.project,
        images_dir=args.images_dir,
        batch_name=args.batch_name
    )


if __name__ == "__main__":
    main()
