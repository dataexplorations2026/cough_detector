#!/usr/bin/env python3
"""Download ESC-50 dataset only (no training)."""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.dataset import download_esc50

if __name__ == '__main__':
    print("Downloading ESC-50 dataset...")
    download_esc50('./datasets')
    print("Done!")
