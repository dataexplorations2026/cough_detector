#!/usr/bin/env python3
"""
Main entry point for real-time cough detection.
Run this script to start listening to the microphone for coughs.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.inference import main

if __name__ == '__main__':
    main()
