#!/usr/bin/env python3
"""
Quick training script - downloads ESC-50 and trains a model.
This is the simplest way to get a working model.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.train import train


def main():
    print("=" * 60)
    print("Cough Detector - Quick Training")
    print("=" * 60)
    print()
    print("This script will:")
    print("1. Download the ESC-50 dataset (~650MB)")
    print("2. Train a cough detection model")
    print("3. Save the model to ./checkpoints/")
    print()
    
    # Train with default settings using ESC-50
    model_path = train(
        data_dir=None,  # No custom data, use ESC-50 only
        output_dir='./checkpoints',
        model_type='small',  # Lightweight model for real-time
        epochs=50,
        batch_size=32,
        learning_rate=0.001,
        patience=10,
        device='auto',
        num_workers=0,  # 0 for Windows compatibility
        use_esc50=True,
        esc50_dir='./datasets'
    )
    
    print()
    print("=" * 60)
    print("Training complete!")
    print("=" * 60)
    print()
    print(f"Model saved to: {model_path}")
    print()
    print("To run live detection:")
    print(f"  python run_detection.py --model {model_path}")
    print()


if __name__ == '__main__':
    main()
