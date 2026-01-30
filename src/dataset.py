"""
Dataset classes for cough detection training.
Supports multiple data sources and automatic downloading.
"""

import os
import json
import torch
import torchaudio
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Callable
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import urllib.request
import zipfile
import tarfile
import shutil

from .preprocessing import AudioPreprocessor
from .augmentation import AudioAugmentor, SpecAugment


class CoughDataset(Dataset):
    """
    Dataset for cough detection.
    
    Loads audio files from a directory structure and prepares them for training.
    Expected structure:
        data_dir/
            cough/
                audio1.wav
                audio2.wav
                ...
            non_cough/
                audio1.wav
                audio2.wav
                ...
    """
    
    def __init__(
        self,
        data_dir: str,
        preprocessor: AudioPreprocessor,
        audio_augmentor: Optional[AudioAugmentor] = None,
        spec_augmentor: Optional[SpecAugment] = None,
        is_training: bool = True,
        cache_spectrograms: bool = False
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Root directory containing class subdirectories
            preprocessor: Audio preprocessor instance
            audio_augmentor: Optional waveform augmentor
            spec_augmentor: Optional spectrogram augmentor
            is_training: Whether this is a training set (enables augmentation)
            cache_spectrograms: Whether to cache computed spectrograms
        """
        self.data_dir = Path(data_dir)
        self.preprocessor = preprocessor
        self.audio_augmentor = audio_augmentor
        self.spec_augmentor = spec_augmentor
        self.is_training = is_training
        self.cache_spectrograms = cache_spectrograms
        
        # Class mapping
        self.classes = ['non_cough', 'cough']
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        # Collect samples
        self.samples = self._collect_samples()
        
        # Spectrogram cache
        self.cache = {} if cache_spectrograms else None
        
        # Calculate class weights for balanced sampling
        self.class_counts = self._count_classes()
        self.sample_weights = self._compute_sample_weights()
    
    def _collect_samples(self) -> List[Tuple[str, int]]:
        """Collect all audio file paths and their labels."""
        samples = []
        audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.webm'}
        
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                print(f"Warning: Class directory {class_dir} not found")
                continue
            
            label = self.class_to_idx[class_name]
            
            for audio_file in class_dir.iterdir():
                if audio_file.suffix.lower() in audio_extensions:
                    samples.append((str(audio_file), label))
        
        return samples
    
    def _count_classes(self) -> Dict[int, int]:
        """Count samples per class."""
        counts = {i: 0 for i in range(len(self.classes))}
        for _, label in self.samples:
            counts[label] += 1
        return counts
    
    def _compute_sample_weights(self) -> torch.Tensor:
        """Compute sample weights for balanced sampling."""
        weights = []
        total = len(self.samples)
        for _, label in self.samples:
            class_weight = total / (len(self.classes) * self.class_counts[label])
            weights.append(class_weight)
        return torch.tensor(weights)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (mel spectrogram, label)
        """
        audio_path, label = self.samples[idx]
        
        # Check cache
        if self.cache is not None and audio_path in self.cache:
            spectrogram = self.cache[audio_path].clone()
            if self.is_training and self.spec_augmentor:
                spectrogram = self.spec_augmentor(spectrogram)
            return spectrogram, label
        
        # Load audio - fail hard if this doesn't work, never silently return zeros
        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load audio file: {audio_path}\n"
                f"Error: {e}\n"
                f"If you see 'torchcodec' errors, run: pip install torchcodec"
            ) from e
        
        # Resample and convert to mono
        waveform = self.preprocessor.resample(waveform, sr)
        waveform = self.preprocessor.to_mono(waveform)
        
        # Apply waveform augmentation during training
        if self.is_training and self.audio_augmentor:
            waveform = self.audio_augmentor.augment(waveform)
        
        # Normalize and pad/trim
        waveform = self.preprocessor.normalize(waveform)
        waveform = self.preprocessor.pad_or_trim(waveform)
        
        # Extract features (mel spectrogram + MFCCs if enabled)
        features = self.preprocessor.extract_features(waveform)
        
        # Cache if enabled (before spec augmentation)
        if self.cache is not None:
            self.cache[audio_path] = features.clone()
        
        # Apply spectrogram augmentation during training
        if self.is_training and self.spec_augmentor:
            features = self.spec_augmentor(features)
        
        return features, label


class ESC50Dataset(Dataset):
    """
    ESC-50 dataset handler.
    
    ESC-50 contains 2000 environmental audio recordings in 50 classes.
    We use the "coughing" class (label 24) and sample from other classes for negatives.
    """
    
    # ESC-50 class that corresponds to cough
    COUGH_CLASS = 24  # "coughing" in ESC-50
    
    # Classes to use as negative examples (similar sounds that could be confused)
    NEGATIVE_CLASSES = [
        20,  # breathing
        21,  # snoring
        22,  # sneezing
        23,  # crying_baby
        25,  # clapping
        26,  # laughing
        38,  # door_knock
    ]
    
    def __init__(
        self,
        data_dir: str,
        preprocessor: AudioPreprocessor,
        audio_augmentor: Optional[AudioAugmentor] = None,
        spec_augmentor: Optional[SpecAugment] = None,
        is_training: bool = True,
        fold: Optional[int] = None,
        include_all_negatives: bool = True
    ):
        """
        Initialize ESC-50 dataset.
        
        Args:
            data_dir: Path to ESC-50 dataset
            preprocessor: Audio preprocessor
            audio_augmentor: Optional waveform augmentor
            spec_augmentor: Optional spectrogram augmentor
            is_training: Whether this is training set
            fold: Specific fold to use (1-5), None for all
            include_all_negatives: Include all non-cough classes as negatives
        """
        self.data_dir = Path(data_dir)
        self.preprocessor = preprocessor
        self.audio_augmentor = audio_augmentor
        self.spec_augmentor = spec_augmentor
        self.is_training = is_training
        self.include_all_negatives = include_all_negatives
        
        # Load metadata
        meta_path = self.data_dir / 'meta' / 'esc50.csv'
        if not meta_path.exists():
            raise FileNotFoundError(f"ESC-50 metadata not found at {meta_path}")
        
        self.metadata = pd.read_csv(meta_path)
        
        # Filter by fold if specified
        if fold is not None:
            if is_training:
                self.metadata = self.metadata[self.metadata['fold'] != fold]
            else:
                self.metadata = self.metadata[self.metadata['fold'] == fold]
        
        # Collect samples
        self.samples = self._collect_samples()
    
    def _collect_samples(self) -> List[Tuple[str, int]]:
        """Collect cough and negative samples."""
        samples = []
        audio_dir = self.data_dir / 'audio'
        
        for _, row in self.metadata.iterrows():
            target = row['target']
            filename = row['filename']
            filepath = audio_dir / filename
            
            if not filepath.exists():
                continue
            
            if target == self.COUGH_CLASS:
                # Cough sample
                samples.append((str(filepath), 1))
            elif self.include_all_negatives or target in self.NEGATIVE_CLASSES:
                # Negative sample
                samples.append((str(filepath), 0))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        audio_path, label = self.samples[idx]
        
        # Load audio - fail hard if this doesn't work, never silently return zeros
        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load audio file: {audio_path}\n"
                f"Error: {e}\n"
                f"If you see 'torchcodec' errors, run: pip install torchcodec"
            ) from e
        
        waveform = self.preprocessor.resample(waveform, sr)
        waveform = self.preprocessor.to_mono(waveform)
        
        if self.is_training and self.audio_augmentor:
            waveform = self.audio_augmentor.augment(waveform)
        
        waveform = self.preprocessor.normalize(waveform)
        waveform = self.preprocessor.pad_or_trim(waveform)
        
        features = self.preprocessor.extract_features(waveform)
        
        if self.is_training and self.spec_augmentor:
            features = self.spec_augmentor(features)
        
        return features, label


class CombinedDataset(Dataset):
    """
    Combines multiple datasets for training.
    """
    
    def __init__(self, datasets: List[Dataset]):
        """
        Initialize combined dataset.
        
        Args:
            datasets: List of datasets to combine
        """
        self.datasets = datasets
        self.cumulative_sizes = []
        
        total = 0
        for ds in datasets:
            total += len(ds)
            self.cumulative_sizes.append(total)
    
    def __len__(self) -> int:
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Find which dataset this index belongs to
        for i, cumsize in enumerate(self.cumulative_sizes):
            if idx < cumsize:
                if i == 0:
                    return self.datasets[i][idx]
                else:
                    return self.datasets[i][idx - self.cumulative_sizes[i-1]]
        raise IndexError(f"Index {idx} out of range")


def download_esc50(target_dir: str) -> str:
    """
    Download ESC-50 dataset.
    
    Args:
        target_dir: Directory to download to
        
    Returns:
        Path to extracted dataset
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    esc50_dir = target_dir / 'ESC-50-master'
    if esc50_dir.exists():
        print("ESC-50 already downloaded")
        return str(esc50_dir)
    
    url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
    zip_path = target_dir / "esc50.zip"
    
    print("Downloading ESC-50 dataset...")
    urllib.request.urlretrieve(url, zip_path)
    
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
    
    # Clean up
    zip_path.unlink()
    
    print(f"ESC-50 downloaded to {esc50_dir}")
    return str(esc50_dir)


def create_data_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    use_weighted_sampler: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        use_weighted_sampler: Use weighted sampler for balanced batches
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create weighted sampler for training if dataset supports it
    sampler = None
    shuffle = True
    
    if use_weighted_sampler and hasattr(train_dataset, 'sample_weights'):
        sampler = WeightedRandomSampler(
            weights=train_dataset.sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        shuffle = False
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def prepare_dataset_split(
    data_dir: str,
    preprocessor: AudioPreprocessor,
    audio_augmentor: Optional[AudioAugmentor] = None,
    spec_augmentor: Optional[SpecAugment] = None,
    val_split: float = 0.2,
    random_state: int = 42
) -> Tuple[CoughDataset, CoughDataset]:
    """
    Prepare train/val split from a single directory.
    
    Args:
        data_dir: Data directory
        preprocessor: Audio preprocessor
        audio_augmentor: Waveform augmentor
        spec_augmentor: Spectrogram augmentor
        val_split: Validation split ratio
        random_state: Random seed
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Create full dataset
    full_dataset = CoughDataset(
        data_dir=data_dir,
        preprocessor=preprocessor,
        is_training=False
    )
    
    # Split indices
    indices = list(range(len(full_dataset)))
    labels = [full_dataset.samples[i][1] for i in indices]
    
    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_split,
        random_state=random_state,
        stratify=labels
    )
    
    # Create subset datasets
    train_samples = [full_dataset.samples[i] for i in train_indices]
    val_samples = [full_dataset.samples[i] for i in val_indices]
    
    # Create new datasets with split samples
    train_dataset = CoughDataset(
        data_dir=data_dir,
        preprocessor=preprocessor,
        audio_augmentor=audio_augmentor,
        spec_augmentor=spec_augmentor,
        is_training=True
    )
    train_dataset.samples = train_samples
    train_dataset.sample_weights = train_dataset._compute_sample_weights()
    
    val_dataset = CoughDataset(
        data_dir=data_dir,
        preprocessor=preprocessor,
        is_training=False
    )
    val_dataset.samples = val_samples
    
    return train_dataset, val_dataset
