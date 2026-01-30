#!/usr/bin/env python3
"""
Train cough detector with balanced dataset.
Run setup_data.py first to prepare the data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.train import train


def main():
    print("=" * 60)
    print("Cough Detector - Training with Balanced Data")
    print("=" * 60)
    
    data_dir = Path("./data")
    
    if not data_dir.exists() or not (data_dir / "cough").exists():
        print("\nData not found! Running setup_data.py first...")
        import setup_data
        setup_data.main()
    
    # Count samples - include all audio formats
    audio_extensions = ['*.wav', '*.webm', '*.ogg', '*.mp3', '*.flac']
    cough_count = sum(len(list((data_dir / "cough").glob(ext))) for ext in audio_extensions)
    non_cough_count = sum(len(list((data_dir / "non_cough").glob(ext))) for ext in audio_extensions)
    
    print(f"\nDataset:")
    print(f"  Cough samples: {cough_count}")
    print(f"  Non-cough samples: {non_cough_count}")
    
    if cough_count == 0:
        print("\nERROR: No cough samples found!")
        print("Make sure ESC-50 is downloaded (run train_quick.py once first)")
        return
    
    # Calculate class weight based on actual ratio
    ratio = non_cough_count / max(cough_count, 1)
    print(f"  Class weight ratio: {ratio:.1f}x")
    
    print("\nStarting training...")
    
    model_path = train(
        data_dir=str(data_dir),
        output_dir='./checkpoints',
        model_type='residual',
        epochs=150,
        batch_size=32,
        learning_rate=0.0005,
        patience=20,
        device='auto',
        num_workers=0,
        use_esc50=False,
        esc50_dir='./datasets'
    )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"\nModel saved to: {model_path}")
    print("\nTo run live detection:")
    print(f"  python run_detection.py --model {model_path}")


if __name__ == "__main__":
    main()
