#!/usr/bin/env python3
"""
Data preparation script.
Downloads and organizes cough detection datasets.
"""

import os
import sys
import json
import urllib.request
import zipfile
import shutil
from pathlib import Path
from typing import Optional
import random

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def download_file(url: str, dest: Path, desc: str = "Downloading"):
    """Download a file with progress."""
    print(f"{desc}: {url}")
    
    def progress_hook(count, block_size, total_size):
        percent = min(100, count * block_size * 100 // total_size)
        sys.stdout.write(f"\r  Progress: {percent}%")
        sys.stdout.flush()
    
    urllib.request.urlretrieve(url, str(dest), progress_hook)
    print()


def download_esc50(output_dir: Path) -> Path:
    """
    Download ESC-50 dataset.
    Contains cough samples and negative samples.
    """
    esc50_dir = output_dir / 'ESC-50-master'
    
    if esc50_dir.exists():
        print("ESC-50 already downloaded")
        return esc50_dir
    
    url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
    zip_path = output_dir / "esc50.zip"
    
    print("\n=== Downloading ESC-50 Dataset ===")
    download_file(url, zip_path, "ESC-50")
    
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    
    zip_path.unlink()
    print(f"ESC-50 extracted to {esc50_dir}")
    
    return esc50_dir


def organize_esc50_for_cough(esc50_dir: Path, output_dir: Path):
    """
    Organize ESC-50 into cough/non_cough structure.
    """
    import pandas as pd
    
    print("\n=== Organizing ESC-50 for Cough Detection ===")
    
    # Create output directories
    cough_dir = output_dir / 'cough'
    non_cough_dir = output_dir / 'non_cough'
    cough_dir.mkdir(parents=True, exist_ok=True)
    non_cough_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    meta_path = esc50_dir / 'meta' / 'esc50.csv'
    df = pd.read_csv(meta_path)
    
    audio_dir = esc50_dir / 'audio'
    
    # ESC-50 classes that are useful for cough detection
    COUGH_CLASS = 24  # coughing
    HARD_NEGATIVE_CLASSES = {
        20: 'breathing',
        21: 'snoring', 
        22: 'sneezing',
        23: 'crying_baby',
        25: 'clapping',
        26: 'laughing',
        38: 'door_knock',
        39: 'mouse_click',
        36: 'vacuum_cleaner'
    }
    
    cough_count = 0
    non_cough_count = 0
    
    for _, row in df.iterrows():
        src = audio_dir / row['filename']
        if not src.exists():
            continue
        
        if row['target'] == COUGH_CLASS:
            dst = cough_dir / row['filename']
            shutil.copy2(src, dst)
            cough_count += 1
        elif row['target'] in HARD_NEGATIVE_CLASSES:
            dst = non_cough_dir / row['filename']
            shutil.copy2(src, dst)
            non_cough_count += 1
    
    print(f"  Cough samples: {cough_count}")
    print(f"  Non-cough samples: {non_cough_count}")
    print(f"  Output directory: {output_dir}")


def create_synthetic_negatives(output_dir: Path, num_samples: int = 100):
    """
    Create synthetic negative samples (silence, white noise, etc.)
    This helps the model distinguish coughs from random noise.
    """
    import numpy as np
    import torch
    import torchaudio
    
    print(f"\n=== Creating {num_samples} Synthetic Negative Samples ===")
    
    non_cough_dir = output_dir / 'non_cough'
    non_cough_dir.mkdir(parents=True, exist_ok=True)
    
    sample_rate = 16000
    duration = 2.0  # 2 seconds
    num_samples_audio = int(sample_rate * duration)
    
    for i in range(num_samples):
        # Randomly select type
        noise_type = random.choice(['silence', 'white_noise', 'pink_noise', 'ambient'])
        
        if noise_type == 'silence':
            # Near-silence with tiny random noise
            audio = np.random.randn(num_samples_audio) * 0.001
        
        elif noise_type == 'white_noise':
            # White noise at low volume
            audio = np.random.randn(num_samples_audio) * random.uniform(0.01, 0.1)
        
        elif noise_type == 'pink_noise':
            # Pink noise (1/f noise)
            white = np.random.randn(num_samples_audio)
            # Simple pink noise approximation using cumulative sum
            pink = np.cumsum(white)
            pink = pink / np.abs(pink).max()
            audio = pink * random.uniform(0.01, 0.1)
        
        else:  # ambient
            # Mix of different frequencies
            t = np.linspace(0, duration, num_samples_audio)
            freqs = [60, 120, 240, 500, 1000]  # Common ambient frequencies
            audio = np.zeros(num_samples_audio)
            for freq in random.sample(freqs, random.randint(1, 3)):
                audio += np.sin(2 * np.pi * freq * t) * random.uniform(0.01, 0.03)
            audio += np.random.randn(num_samples_audio) * 0.005
        
        # Convert to tensor and save
        audio = audio.astype(np.float32)
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        
        filename = f"synthetic_{noise_type}_{i:04d}.wav"
        torchaudio.save(str(non_cough_dir / filename), audio_tensor, sample_rate)
    
    print(f"  Created {num_samples} synthetic samples")


def print_dataset_stats(data_dir: Path):
    """Print statistics about the prepared dataset."""
    print("\n=== Dataset Statistics ===")
    
    for class_name in ['cough', 'non_cough']:
        class_dir = data_dir / class_name
        if class_dir.exists():
            files = list(class_dir.glob('*.wav')) + list(class_dir.glob('*.ogg'))
            print(f"  {class_name}: {len(files)} samples")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare cough detection datasets')
    parser.add_argument('--output-dir', type=str, default='./data',
                        help='Output directory for prepared data')
    parser.add_argument('--datasets-dir', type=str, default='./datasets',
                        help='Directory to download raw datasets')
    parser.add_argument('--synthetic-samples', type=int, default=100,
                        help='Number of synthetic negative samples to create')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    datasets_dir = Path(args.datasets_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Cough Detection - Data Preparation")
    print("=" * 60)
    
    # Download and organize ESC-50
    esc50_dir = download_esc50(datasets_dir)
    organize_esc50_for_cough(esc50_dir, output_dir)
    
    # Create synthetic negatives
    create_synthetic_negatives(output_dir, args.synthetic_samples)
    
    # Print statistics
    print_dataset_stats(output_dir)
    
    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)
    print(f"\nPrepared data is in: {output_dir}")
    print("\nTo train with this data:")
    print(f"  python src/train.py --data-dir {output_dir} --no-esc50")
    print()


if __name__ == '__main__':
    main()
