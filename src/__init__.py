"""
Cough Detection ML Pipeline

A real-time machine learning pipeline for detecting coughs from microphone input.
"""

__version__ = "1.0.0"
__author__ = "Cough Detector Team"

from .preprocessing import AudioPreprocessor, RealtimePreprocessor, create_preprocessor
from .model import CoughDetector, CoughDetectorSmall, CoughDetectorResidual, create_model
from .augmentation import AudioAugmentor, SpecAugment, create_augmentation_pipeline
from .dataset import CoughDataset, ESC50Dataset, download_esc50

__all__ = [
    'AudioPreprocessor',
    'RealtimePreprocessor', 
    'create_preprocessor',
    'CoughDetector',
    'CoughDetectorSmall',
    'CoughDetectorResidual',
    'create_model',
    'AudioAugmentor',
    'SpecAugment',
    'create_augmentation_pipeline',
    'CoughDataset',
    'ESC50Dataset',
    'download_esc50'
]
