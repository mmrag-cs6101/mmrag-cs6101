#!/usr/bin/env python3
"""
MRAG-Bench Dataset Download Script

Downloads and organizes the MRAG-Bench dataset from HuggingFace.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any
from datasets import load_dataset
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_mrag_bench_dataset(data_dir: str = "data/mrag_bench") -> Dict[str, Any]:
    """
    Download and organize MRAG-Bench dataset.

    Args:
        data_dir: Directory to save the dataset

    Returns:
        Dictionary with dataset statistics
    """
    data_path = Path(data_dir)

    # Create directory structure
    (data_path / "images").mkdir(parents=True, exist_ok=True)
    (data_path / "questions").mkdir(parents=True, exist_ok=True)
    (data_path / "annotations").mkdir(parents=True, exist_ok=True)
    (data_path / "metadata").mkdir(parents=True, exist_ok=True)
    (data_path / "analysis").mkdir(parents=True, exist_ok=True)

    logger.info("Downloading MRAG-Bench dataset from HuggingFace...")

    try:
        # Load dataset
        dataset = load_dataset('uclanlp/MRAG-Bench', split='test')
        logger.info(f"Dataset loaded: {len(dataset)} samples")

        # Analyze dataset structure
        sample = dataset[0]
        logger.info(f"Dataset features: {list(dataset.features.keys())}")

        # Save sample for analysis
        sample_data = []
        questions_data = []
        metadata_info = {
            "total_samples": len(dataset),
            "features": list(dataset.features.keys()),
            "scenarios": {},
            "image_count": 0
        }

        # Process each sample
        for i, item in enumerate(dataset):
            # Save image
            if 'image' in item and item['image'] is not None:
                image_filename = f"image_{i:06d}.jpg"
                image_path = data_path / "images" / image_filename

                # Convert RGBA to RGB if necessary for JPEG saving
                image = item['image']
                if image.mode == 'RGBA':
                    # Create white background for transparency
                    rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                    rgb_image.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
                    image = rgb_image
                elif image.mode not in ['RGB', 'L']:
                    # Convert other modes to RGB
                    image = image.convert('RGB')

                image.save(image_path, 'JPEG', quality=95)
                metadata_info["image_count"] += 1

            # Prepare question data
            question_item = {
                "question_id": f"mrag_{i:06d}",
                "question": item.get('question', ''),
                "choices": item.get('choices', []),
                "answer": item.get('answer', ''),
                "image_path": f"images/{image_filename}" if 'image' in item else "",
                "category": item.get('category', 'unknown'),
                "scenario": item.get('scenario', item.get('category', 'unknown'))
            }

            questions_data.append(question_item)

            # Track scenarios
            scenario = question_item['scenario']
            if scenario not in metadata_info["scenarios"]:
                metadata_info["scenarios"][scenario] = 0
            metadata_info["scenarios"][scenario] += 1

            # Save sample data for first 10 items
            if i < 10:
                sample_item = {}
                for key, value in item.items():
                    if key == 'image':
                        sample_item[key] = f"<PIL_Image_{i}>" if value is not None else None
                    elif key == 'gt_images':
                        sample_item[key] = f"<PIL_Images_list_{len(value) if value else 0}>" if value else None
                    elif key == 'retrieved_images':
                        sample_item[key] = f"<PIL_Images_list_{len(value) if value else 0}>" if value else None
                    else:
                        # Handle other non-serializable objects
                        try:
                            json.dumps(value)  # Test if serializable
                            sample_item[key] = value
                        except (TypeError, ValueError):
                            sample_item[key] = f"<{type(value).__name__}>"
                sample_data.append(sample_item)

            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(dataset)} samples")

        # Save processed data
        logger.info("Saving processed data...")

        # Save questions
        with open(data_path / "questions" / "questions.json", 'w') as f:
            json.dump(questions_data, f, indent=2)

        # Save metadata
        with open(data_path / "metadata" / "dataset_info.json", 'w') as f:
            json.dump(metadata_info, f, indent=2)

        # Save sample data for analysis
        with open(data_path / "analysis" / "sample_data.json", 'w') as f:
            json.dump(sample_data, f, indent=2)

        # Create scenario mapping
        scenario_mapping = {
            scenario: {
                "count": count,
                "samples": [q for q in questions_data if q['scenario'] == scenario]
            }
            for scenario, count in metadata_info["scenarios"].items()
        }

        with open(data_path / "metadata" / "scenario_mapping.json", 'w') as f:
            json.dump(scenario_mapping, f, indent=2)

        logger.info(f"Dataset download and organization complete!")
        logger.info(f"Total samples: {metadata_info['total_samples']}")
        logger.info(f"Total images: {metadata_info['image_count']}")
        logger.info(f"Scenarios: {list(metadata_info['scenarios'].keys())}")

        return metadata_info

    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        raise

if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/mrag_bench"
    result = download_mrag_bench_dataset(data_dir)
    print(f"Download complete. Statistics: {result}")