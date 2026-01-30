#!/usr/bin/env python3
"""
Download and prepare COUGHVID dataset for cough detection training.

COUGHVID contains ~25,000 crowdsourced cough recordings.
We filter for high-quality coughs and balance with non-cough samples.

Source: https://zenodo.org/records/7024894
Paper: https://www.nature.com/articles/s41597-021-00937-4
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
    print(f"\n{desc}")
    print(f"URL: {url}")
    print(f"Destination: {dest}")
    
    try:
        def progress_hook(count, block_size, total_size):
            if total_size > 0:
                percent = min(100, count * block_size * 100 // total_size)
                downloaded_mb = count * block_size / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                sys.stdout.write(f"\rProgress: {percent}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, str(dest), progress_hook)
        print("\nDownload complete!")
        return True
    except Exception as e:
        print(f"\nDownload failed: {e}")
        return False


def download_coughvid(datasets_dir: Path):
    """
    Download COUGHVID dataset from Zenodo.
    """
    datasets_dir = Path(datasets_dir)
    datasets_dir.mkdir(parents=True, exist_ok=True)
    
    coughvid_dir = datasets_dir / "coughvid"
    
    if coughvid_dir.exists() and len(list(coughvid_dir.glob("*.webm"))) > 1000:
        print("COUGHVID already downloaded")
        return coughvid_dir
    
    coughvid_dir.mkdir(parents=True, exist_ok=True)
    
    # COUGHVID public dataset on Zenodo (version 1.0)
    # Contains ~20,000 recordings with metadata
    print("\n" + "=" * 60)
    print("Downloading COUGHVID Dataset")
    print("=" * 60)
    print("\nThis dataset contains crowdsourced cough recordings.")
    print("Size: ~950 MB")
    print("Source: https://zenodo.org/records/4048312")
    
    # Correct URL for public_dataset.zip (version 1.0)
    zip_url = "https://zenodo.org/records/4048312/files/public_dataset.zip?download=1"
    zip_path = datasets_dir / "coughvid.zip"
    
    if not zip_path.exists():
        success = download_file(zip_url, zip_path, "Downloading COUGHVID dataset...")
        if not success:
            print("\nFailed to download COUGHVID.")
            print("You can manually download from: https://zenodo.org/records/7024894")
            print(f"Extract to: {coughvid_dir}")
            return None
    
    # Extract
    print("\nExtracting...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(coughvid_dir)
        print("Extraction complete!")
    except Exception as e:
        print(f"Extraction failed: {e}")
        return None
    
    # Clean up zip file to save space
    # zip_path.unlink()
    
    return coughvid_dir


def prepare_coughvid_data(coughvid_dir: Path, output_dir: Path, max_coughs: int = 2000):
    """
    Process COUGHVID dataset and prepare for training.
    
    Args:
        coughvid_dir: Path to extracted COUGHVID dataset
        output_dir: Output directory for processed data
        max_coughs: Maximum number of cough samples to use
    """
    import pandas as pd
    import numpy as np
    
    print("\n" + "=" * 60)
    print("Processing COUGHVID Dataset")
    print("=" * 60)
    
    output_dir = Path(output_dir)
    cough_dir = output_dir / "cough"
    non_cough_dir = output_dir / "non_cough"
    
    cough_dir.mkdir(parents=True, exist_ok=True)
    non_cough_dir.mkdir(parents=True, exist_ok=True)
    
    # Find metadata file
    metadata_files = list(coughvid_dir.rglob("*.csv"))
    if not metadata_files:
        print("ERROR: No metadata CSV found in COUGHVID directory")
        return 0, 0
    
    metadata_path = metadata_files[0]
    print(f"\nLoading metadata from: {metadata_path}")
    
    df = pd.read_csv(metadata_path)
    print(f"Total recordings in metadata: {len(df)}")
    
    # Print available columns
    print(f"Available columns: {list(df.columns)}")
    
    # Find audio files
    audio_extensions = ['.webm', '.ogg', '.wav', '.mp3']
    audio_dir = coughvid_dir
    
    # Look for audio files in subdirectories too
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(list(coughvid_dir.rglob(f"*{ext}")))
    
    print(f"Found {len(audio_files)} audio files")
    
    if len(audio_files) == 0:
        print("ERROR: No audio files found!")
        return 0, 0
    
    # Create a mapping from UUID to file path
    uuid_to_file = {}
    for f in audio_files:
        uuid = f.stem
        uuid_to_file[uuid] = f
    
    # Filter for high-quality cough recordings
    # COUGHVID has 'cough_detected' probability and 'status' column
    cough_samples = []
    non_cough_samples = []
    
    # Check which columns exist for filtering
    has_cough_detected = 'cough_detected' in df.columns
    has_status = 'status' in df.columns
    has_quality = 'quality' in df.columns
    
    print(f"\nFiltering columns available:")
    print(f"  cough_detected: {has_cough_detected}")
    print(f"  status: {has_status}")
    print(f"  quality: {has_quality}")
    
    for _, row in df.iterrows():
        uuid = str(row.get('uuid', row.get('filename', ''))).replace('.webm', '').replace('.ogg', '')
        
        if uuid not in uuid_to_file:
            continue
        
        filepath = uuid_to_file[uuid]
        
        # Determine if this is a cough based on available columns
        is_cough = False
        is_non_cough = False
        cough_confidence = 0.0
        
        if has_cough_detected:
            cough_prob = row.get('cough_detected', 0)
            if pd.notna(cough_prob):
                cough_confidence = float(cough_prob)
                # Use high-confidence coughs (>0.8 instead of >0.9 for more data)
                if cough_confidence > 0.8:
                    is_cough = True
                elif cough_confidence < 0.2:
                    is_non_cough = True
        
        if has_status:
            status = str(row.get('status', '')).lower()
            if 'healthy' in status and cough_confidence < 0.5:
                is_non_cough = True
        
        if is_cough:
            cough_samples.append((filepath, cough_confidence))
        elif is_non_cough:
            non_cough_samples.append(filepath)
    
    print(f"\nFiltered samples:")
    print(f"  High-confidence coughs: {len(cough_samples)}")
    print(f"  Non-coughs: {len(non_cough_samples)}")
    
    # If filtering didn't work well, use a simpler approach
    if len(cough_samples) < 100:
        print("\nNot enough filtered coughs, using alternative approach...")
        # Just take all samples and assume most are coughs (it's a cough dataset after all)
        all_files = list(audio_files)
        np.random.shuffle(all_files)
        
        # Take 80% as coughs, 20% as non-coughs (many "failed" recordings are just noise)
        split_point = int(len(all_files) * 0.8)
        cough_samples = [(f, 0.5) for f in all_files[:split_point]]
        non_cough_samples = all_files[split_point:]
        
        print(f"  Using {len(cough_samples)} as coughs")
        print(f"  Using {len(non_cough_samples)} as non-coughs")
    
    # Sort by confidence and take the best ones
    cough_samples.sort(key=lambda x: x[1], reverse=True)
    
    # Limit samples - take highest confidence coughs
    if len(cough_samples) > max_coughs:
        cough_samples = cough_samples[:max_coughs]
    
    # Extract just file paths
    cough_files = [f[0] for f in cough_samples]
    
    # Balance non-coughs to be roughly equal or slightly more
    max_non_coughs = int(len(cough_files) * 1.5)
    if len(non_cough_samples) > max_non_coughs:
        np.random.shuffle(non_cough_samples)
        non_cough_samples = non_cough_samples[:max_non_coughs]
    
    # Copy files and convert webm to wav
    print(f"\nCopying and converting {len(cough_files)} cough samples...")
    cough_count = 0
    for i, src in enumerate(cough_files):
        try:
            # Convert webm/ogg to wav using ffmpeg
            if src.suffix.lower() in ['.webm', '.ogg']:
                dst = cough_dir / f"coughvid_{i:04d}.wav"
                # Use ffmpeg to convert
                import subprocess
                result = subprocess.run(
                    ['ffmpeg', '-y', '-i', str(src), '-ar', '16000', '-ac', '1', str(dst)],
                    capture_output=True,
                    timeout=30
                )
                if result.returncode == 0:
                    cough_count += 1
            else:
                dst = cough_dir / f"coughvid_{i:04d}{src.suffix}"
                shutil.copy2(src, dst)
                cough_count += 1
            
            if (i + 1) % 200 == 0:
                print(f"  Processed {i + 1}/{len(cough_files)}")
        except Exception as e:
            pass
    
    print(f"\nCopying and converting {len(non_cough_samples)} non-cough samples...")
    non_cough_count = 0
    for i, src in enumerate(non_cough_samples):
        try:
            # Convert webm/ogg to wav using ffmpeg
            if src.suffix.lower() in ['.webm', '.ogg']:
                dst = non_cough_dir / f"coughvid_other_{i:04d}.wav"
                import subprocess
                result = subprocess.run(
                    ['ffmpeg', '-y', '-i', str(src), '-ar', '16000', '-ac', '1', str(dst)],
                    capture_output=True,
                    timeout=30
                )
                if result.returncode == 0:
                    non_cough_count += 1
            else:
                dst = non_cough_dir / f"coughvid_other_{i:04d}{src.suffix}"
                shutil.copy2(src, dst)
                non_cough_count += 1
            
            if (i + 1) % 200 == 0:
                print(f"  Processed {i + 1}/{len(non_cough_samples)}")
        except Exception as e:
            pass
    
    return cough_count, non_cough_count


def add_esc50_data(esc50_dir: Path, output_dir: Path):
    """Add ESC-50 cough and hard negative samples."""
    import pandas as pd
    
    print("\n" + "=" * 60)
    print("Adding ESC-50 Dataset (Hard Negatives)")
    print("=" * 60)
    
    esc50_dir = Path(esc50_dir)
    output_dir = Path(output_dir)
    
    cough_dir = output_dir / "cough"
    non_cough_dir = output_dir / "non_cough"
    
    if not esc50_dir.exists():
        print("ESC-50 not found, skipping...")
        return 0, 0
    
    meta_path = esc50_dir / "meta" / "esc50.csv"
    audio_dir = esc50_dir / "audio"
    
    if not meta_path.exists():
        print("ESC-50 metadata not found, skipping...")
        return 0, 0
    
    df = pd.read_csv(meta_path)
    
    COUGH_CLASS = 24
    
    # Hard negatives - sounds that could be confused with coughs
    HARD_NEGATIVES = [
        20,  # crying_baby
        21,  # sneezing
        22,  # clapping
        23,  # breathing
        25,  # footsteps
        26,  # laughing
        27,  # brushing_teeth
        28,  # snoring
        29,  # drinking_sipping
        30,  # door_knock
        31,  # mouse_click
        32,  # keyboard_typing
        34,  # can_opening
        38,  # clock_alarm
        0,   # dog bark
        35,  # washing_machine
        36,  # vacuum_cleaner
    ]
    
    cough_count = 0
    non_cough_count = 0
    
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
    
    print(f"Added {cough_count} ESC-50 coughs")
    print(f"Added {non_cough_count} ESC-50 hard negatives")
    
    return cough_count, non_cough_count


def generate_synthetic_samples(output_dir: Path, num_coughs: int = 100, num_others: int = 200):
    """Generate synthetic training samples."""
    import numpy as np
    import torch
    import torchaudio
    
    print("\n" + "=" * 60)
    print("Generating Synthetic Samples")
    print("=" * 60)
    
    output_dir = Path(output_dir)
    cough_dir = output_dir / "cough"
    non_cough_dir = output_dir / "non_cough"
    
    sample_rate = 16000
    
    # Generate cough-like sounds
    print(f"Generating {num_coughs} synthetic cough sounds...")
    for i in range(num_coughs):
        duration = np.random.uniform(0.3, 0.8)
        num_samples = int(sample_rate * 2.0)
        t = np.linspace(0, 2.0, num_samples)
        
        cough_start = np.random.uniform(0.3, 1.0)
        envelope = np.zeros(num_samples)
        start_idx = int(cough_start * sample_rate)
        cough_samples = int(duration * sample_rate)
        
        if start_idx + cough_samples < num_samples:
            attack = np.linspace(0, 1, int(0.02 * sample_rate))
            decay = np.exp(-np.linspace(0, 5, cough_samples - len(attack)))
            cough_env = np.concatenate([attack, decay])
            envelope[start_idx:start_idx + len(cough_env)] = cough_env[:min(len(cough_env), num_samples - start_idx)]
        
        noise = np.random.randn(num_samples)
        low_freq = np.sin(2 * np.pi * np.random.uniform(80, 150) * t)
        mid_freq = np.sin(2 * np.pi * np.random.uniform(200, 400) * t)
        
        audio = envelope * (0.7 * noise + 0.2 * low_freq + 0.1 * mid_freq)
        audio = audio / (np.abs(audio).max() + 1e-8) * 0.8
        audio += np.random.randn(num_samples) * 0.01
        
        audio_tensor = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        torchaudio.save(str(cough_dir / f"synthetic_cough_{i:04d}.wav"), audio_tensor, sample_rate)
    
    # Generate non-cough sounds
    print(f"Generating {num_others} synthetic non-cough sounds...")
    for i in range(num_others):
        num_samples = int(sample_rate * 2.0)
        t = np.linspace(0, 2.0, num_samples)
        
        sound_type = np.random.choice(['silence', 'white_noise', 'hum', 'clicks', 'speech_like'])
        
        if sound_type == 'silence':
            audio = np.random.randn(num_samples) * 0.005
        elif sound_type == 'white_noise':
            audio = np.random.randn(num_samples) * np.random.uniform(0.02, 0.1)
        elif sound_type == 'hum':
            freq = np.random.choice([50, 60, 100, 120])
            audio = np.sin(2 * np.pi * freq * t) * 0.1
            audio += np.random.randn(num_samples) * 0.02
        elif sound_type == 'clicks':
            audio = np.random.randn(num_samples) * 0.01
            for _ in range(np.random.randint(1, 5)):
                click_pos = np.random.randint(0, num_samples - 100)
                audio[click_pos:click_pos + 50] = np.random.uniform(-0.3, 0.3)
        else:  # speech_like
            # Multiple formant-like frequencies
            audio = np.zeros(num_samples)
            for _ in range(np.random.randint(2, 5)):
                freq = np.random.uniform(100, 1000)
                audio += np.sin(2 * np.pi * freq * t) * np.random.uniform(0.05, 0.15)
            audio += np.random.randn(num_samples) * 0.02
        
        audio = audio / (np.abs(audio).max() + 1e-8) * 0.5
        audio_tensor = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        torchaudio.save(str(non_cough_dir / f"synthetic_other_{i:04d}.wav"), audio_tensor, sample_rate)
    
    print("Synthetic generation complete!")
    return num_coughs, num_others


def main():
    print("=" * 60)
    print("COUGHVID Dataset Setup (Improved)")
    print("=" * 60)
    print("\nThis script will:")
    print("1. Download COUGHVID dataset (~1.5 GB)")
    print("2. Process and filter for high-quality coughs (3000+)")
    print("3. Add ESC-50 coughs and ALL hard negatives")
    print("4. NO synthetic data - real recordings only")
    print("5. Create balanced training dataset")
    
    datasets_dir = Path("./datasets")
    output_dir = Path("./data")
    
    # Clean output directory
    if output_dir.exists():
        print(f"\nClearing existing data directory: {output_dir}")
        shutil.rmtree(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cough").mkdir(exist_ok=True)
    (output_dir / "non_cough").mkdir(exist_ok=True)
    
    total_coughs = 0
    total_non_coughs = 0
    
    # Download and process COUGHVID
    coughvid_dir = download_coughvid(datasets_dir)
    if coughvid_dir:
        c, nc = prepare_coughvid_data(coughvid_dir, output_dir, max_coughs=3000)
        total_coughs += c
        total_non_coughs += nc
    
    # Add ESC-50 data
    esc50_dir = datasets_dir / "ESC-50-master"
    c, nc = add_esc50_data(esc50_dir, output_dir)
    total_coughs += c
    total_non_coughs += nc
    
    # NO synthetic samples - real data only
    # c, nc = generate_synthetic_samples(output_dir, num_coughs=100, num_others=200)
    
    # Final count - include all audio formats
    audio_extensions = ['*.wav', '*.webm', '*.ogg', '*.mp3', '*.flac']
    final_coughs = sum(len(list((output_dir / "cough").glob(ext))) for ext in audio_extensions)
    final_non_coughs = sum(len(list((output_dir / "non_cough").glob(ext))) for ext in audio_extensions)
    
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Cough samples:     {final_coughs}")
    print(f"Non-cough samples: {final_non_coughs}")
    print(f"Total:             {final_coughs + final_non_coughs}")
    print(f"Ratio:             1:{final_non_coughs/max(final_coughs,1):.1f}")
    print(f"Output directory:  {output_dir}")
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("\nTo train with this data:")
    print("  python train_with_data.py")
    

if __name__ == "__main__":
    main()
