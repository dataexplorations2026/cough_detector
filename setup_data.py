#!/usr/bin/env python3
"""
Download additional cough datasets to improve model training.
"""

import os
import sys
import urllib.request
import zipfile
import shutil
from pathlib import Path
import json

def download_file(url: str, dest: Path, desc: str = "Downloading"):
    """Download a file with progress."""
    print(f"{desc}...")
    print(f"  URL: {url}")
    
    try:
        def progress_hook(count, block_size, total_size):
            if total_size > 0:
                percent = min(100, count * block_size * 100 // total_size)
                sys.stdout.write(f"\r  Progress: {percent}%")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, str(dest), progress_hook)
        print()
        return True
    except Exception as e:
        print(f"\n  Failed: {e}")
        return False


def setup_cough_data(output_dir: Path):
    """
    Set up cough training data by combining multiple sources.
    """
    output_dir = Path(output_dir)
    cough_dir = output_dir / "cough"
    non_cough_dir = output_dir / "non_cough"
    
    cough_dir.mkdir(parents=True, exist_ok=True)
    non_cough_dir.mkdir(parents=True, exist_ok=True)
    
    cough_count = 0
    non_cough_count = 0
    
    # Source 1: ESC-50 dataset
    print("\n=== Processing ESC-50 Dataset ===")
    esc50_dir = Path("./datasets/ESC-50-master")
    
    if esc50_dir.exists():
        import pandas as pd
        meta_path = esc50_dir / "meta" / "esc50.csv"
        audio_dir = esc50_dir / "audio"
        
        if meta_path.exists():
            df = pd.read_csv(meta_path)
            
            # ESC-50 class mappings
            COUGH_CLASS = 24
            # Hard negatives - sounds that might be confused with coughs
            HARD_NEGATIVES = [20, 21, 22, 23, 25, 26, 38]  # breathing, snoring, sneezing, crying, clapping, laughing, door knock
            
            for _, row in df.iterrows():
                src = audio_dir / row['filename']
                if not src.exists():
                    continue
                
                if row['target'] == COUGH_CLASS:
                    dst = cough_dir / f"esc50_{row['filename']}"
                    shutil.copy2(src, dst)
                    cough_count += 1
                elif row['target'] in HARD_NEGATIVES:
                    dst = non_cough_dir / f"esc50_{row['filename']}"
                    shutil.copy2(src, dst)
                    non_cough_count += 1
            
            print(f"  ESC-50 coughs: {cough_count}")
    else:
        print("  ESC-50 not found - run train_quick.py first to download it")
    
    # Source 2: Generate synthetic cough-like sounds for augmentation
    print("\n=== Generating Synthetic Training Data ===")
    
    try:
        import numpy as np
        import torch
        import torchaudio
        
        sample_rate = 16000
        
        # Generate varied "cough-like" synthetic sounds
        # These help the model learn the general pattern even if not perfect coughs
        for i in range(50):
            duration = np.random.uniform(0.3, 0.8)  # Coughs are typically 0.3-0.8 seconds
            num_samples = int(sample_rate * 2.0)  # 2 second clips
            
            # Create a cough-like envelope (sharp attack, quick decay)
            t = np.linspace(0, 2.0, num_samples)
            
            # Random start time for the "cough"
            cough_start = np.random.uniform(0.3, 1.0)
            cough_duration = duration
            
            # Envelope: sharp attack, exponential decay
            envelope = np.zeros(num_samples)
            start_idx = int(cough_start * sample_rate)
            cough_samples = int(cough_duration * sample_rate)
            
            if start_idx + cough_samples < num_samples:
                attack = np.linspace(0, 1, int(0.02 * sample_rate))  # 20ms attack
                decay = np.exp(-np.linspace(0, 5, cough_samples - len(attack)))
                cough_env = np.concatenate([attack, decay])
                envelope[start_idx:start_idx + len(cough_env)] = cough_env[:min(len(cough_env), num_samples - start_idx)]
            
            # Cough-like noise burst (broadband with some formants)
            noise = np.random.randn(num_samples)
            
            # Add some low-frequency components (chest resonance)
            low_freq = np.sin(2 * np.pi * np.random.uniform(80, 150) * t)
            mid_freq = np.sin(2 * np.pi * np.random.uniform(200, 400) * t)
            
            # Combine
            audio = envelope * (0.7 * noise + 0.2 * low_freq + 0.1 * mid_freq)
            audio = audio / (np.abs(audio).max() + 1e-8) * 0.8
            
            # Add slight background noise
            audio += np.random.randn(num_samples) * 0.01
            
            audio_tensor = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
            torchaudio.save(str(cough_dir / f"synthetic_cough_{i:03d}.wav"), audio_tensor, sample_rate)
            cough_count += 1
        
        print(f"  Generated 50 synthetic cough samples")
        
        # Generate non-cough sounds (silence, noise, speech-like)
        for i in range(100):
            num_samples = int(sample_rate * 2.0)
            t = np.linspace(0, 2.0, num_samples)
            
            sound_type = np.random.choice(['silence', 'white_noise', 'hum', 'clicks'])
            
            if sound_type == 'silence':
                audio = np.random.randn(num_samples) * 0.005
            elif sound_type == 'white_noise':
                audio = np.random.randn(num_samples) * np.random.uniform(0.02, 0.1)
            elif sound_type == 'hum':
                freq = np.random.choice([50, 60, 100, 120])  # electrical hum
                audio = np.sin(2 * np.pi * freq * t) * 0.1
                audio += np.random.randn(num_samples) * 0.02
            else:  # clicks
                audio = np.random.randn(num_samples) * 0.01
                # Add random clicks
                for _ in range(np.random.randint(1, 5)):
                    click_pos = np.random.randint(0, num_samples - 100)
                    audio[click_pos:click_pos + 50] = np.random.uniform(-0.3, 0.3)
            
            audio = audio / (np.abs(audio).max() + 1e-8) * 0.5
            audio_tensor = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
            torchaudio.save(str(non_cough_dir / f"synthetic_other_{i:03d}.wav"), audio_tensor, sample_rate)
            non_cough_count += 1
        
        print(f"  Generated 100 synthetic non-cough samples")
        
    except Exception as e:
        print(f"  Could not generate synthetic data: {e}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    
    final_cough = len(list(cough_dir.glob("*.wav")))
    final_non_cough = len(list(non_cough_dir.glob("*.wav")))
    
    print(f"Cough samples:     {final_cough}")
    print(f"Non-cough samples: {final_non_cough}")
    print(f"Total:             {final_cough + final_non_cough}")
    print(f"Output directory:  {output_dir}")
    
    return final_cough, final_non_cough


def main():
    print("=" * 60)
    print("Cough Detection - Data Setup")
    print("=" * 60)
    
    output_dir = Path("./data")
    
    cough_count, non_cough_count = setup_cough_data(output_dir)
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("\nTo train with this data, run:")
    print("  python train_with_data.py")
    print()


if __name__ == "__main__":
    main()
