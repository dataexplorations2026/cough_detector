"""
Training script for cough detection model.
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import AudioPreprocessor
from src.augmentation import AudioAugmentor, SpecAugment
from src.model import create_model, count_parameters
from src.dataset import (
    CoughDataset, ESC50Dataset, CombinedDataset,
    create_data_loaders, download_esc50, prepare_dataset_split
)


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        
    Returns:
        Dictionary of training metrics
    """
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    return {
        'loss': running_loss / len(train_loader),
        'accuracy': 100. * correct / total
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Validate the model.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device
        
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # For detailed metrics
    all_preds = []
    all_targets = []
    
    for inputs, targets in tqdm(val_loader, desc="Validation"):
        inputs, targets = inputs.to(device), targets.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
    
    # Calculate precision, recall, F1 for cough class
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    tp = ((all_preds == 1) & (all_targets == 1)).sum()
    fp = ((all_preds == 1) & (all_targets == 0)).sum()
    fn = ((all_preds == 0) & (all_targets == 1)).sum()
    tn = ((all_preds == 0) & (all_targets == 0)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'loss': running_loss / len(val_loader),
        'accuracy': 100. * correct / total,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn)
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: Dict,
    path: str,
    config: Dict
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None
) -> Tuple[int, Dict]:
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['metrics']


def train(
    data_dir: str,
    output_dir: str,
    model_type: str = 'small',
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    weight_decay: float = 0.01,
    patience: int = 15,
    device: str = 'auto',
    num_workers: int = 4,
    resume: Optional[str] = None,
    use_esc50: bool = True,
    esc50_dir: Optional[str] = None
):
    """
    Main training function.
    
    Args:
        data_dir: Directory containing cough/non_cough subdirectories
        output_dir: Directory to save checkpoints and logs
        model_type: Model architecture ('standard', 'small', 'residual')
        epochs: Maximum number of epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        weight_decay: Weight decay for regularization
        patience: Early stopping patience
        device: Device to train on ('auto', 'cpu', 'cuda', 'mps')
        num_workers: Data loader workers
        resume: Path to checkpoint to resume from
        use_esc50: Whether to include ESC-50 dataset
        esc50_dir: Path to ESC-50 dataset (will download if not exists)
    """
    # Setup device
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    device = torch.device(device)
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration with enhanced preprocessing
    config = {
        'model_type': model_type,
        'sample_rate': 16000,
        'n_mels': 64,
        'n_fft': 512,
        'hop_length': 160,
        'win_length': 400,
        'f_min': 100.0,  # Bandpass lower bound
        'f_max': 4000.0,  # Bandpass upper bound
        'segment_duration': 1.0,
        'n_mfcc': 13,
        'use_mfcc': True,
        'use_pcen': False,
        'use_pre_emphasis': False,
        'pre_emphasis_coef': 0.97,
        'use_delta_delta': False,
        'use_spectral_contrast': False,
        'n_contrast_bands': 6,
        'batch_size': batch_size,
        'learning_rate': 0.0005,
        'weight_decay': weight_decay,
        'epochs': 150,
        'patience': 20
    }
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create preprocessor with enhanced features
    preprocessor = AudioPreprocessor(
        sample_rate=config['sample_rate'],
        n_mels=config['n_mels'],
        n_fft=config['n_fft'],
        hop_length=config['hop_length'],
        win_length=config['win_length'],
        f_min=config['f_min'],
        f_max=config['f_max'],
        segment_duration=config['segment_duration'],
        n_mfcc=config['n_mfcc'],
        use_mfcc=config['use_mfcc'],
        use_pcen=config['use_pcen'],
        use_pre_emphasis=config['use_pre_emphasis'],
        pre_emphasis_coef=config['pre_emphasis_coef'],
        use_delta_delta=config['use_delta_delta'],
        use_spectral_contrast=config['use_spectral_contrast'],
        n_contrast_bands=config['n_contrast_bands'],
        device='cpu'  # Preprocessing on CPU, model on device
    )
    
    print(f"Preprocessing: Bandpass {config['f_min']}-{config['f_max']}Hz")
    print(f"  Pre-emphasis: {config['use_pre_emphasis']}, PCEN: {config['use_pcen']}")
    print(f"  MFCC: {config['use_mfcc']}, Delta-delta: {config['use_delta_delta']}")
    print(f"  Spectral contrast: {config['use_spectral_contrast']}")
    
    # Enable safe augmentation (speed perturbation disabled due to memory issues)
    audio_augmentor = AudioAugmentor(
        sample_rate=config['sample_rate'],
        p_augment=0.3  # Lower probability for stability
    )
    spec_augmentor = SpecAugment(
        freq_mask_param=8,
        time_mask_param=15,
        n_freq_masks=2,
        n_time_masks=2,
        p=0.3
    )
    
    # Prepare datasets
    datasets_train = []
    datasets_val = []
    
    # Custom dataset
    if data_dir and Path(data_dir).exists():
        print(f"Loading custom dataset from {data_dir}")
        train_ds, val_ds = prepare_dataset_split(
            data_dir=data_dir,
            preprocessor=preprocessor,
            audio_augmentor=audio_augmentor,
            spec_augmentor=spec_augmentor,
            val_split=0.2
        )
        datasets_train.append(train_ds)
        datasets_val.append(val_ds)
        print(f"  Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    
    # ESC-50 dataset
    if use_esc50:
        if esc50_dir is None:
            esc50_dir = str(Path(data_dir).parent / 'datasets' if data_dir else './datasets')
        
        esc50_path = download_esc50(esc50_dir)
        
        # Use fold 5 for validation
        esc50_train = ESC50Dataset(
            data_dir=esc50_path,
            preprocessor=preprocessor,
            audio_augmentor=audio_augmentor,
            spec_augmentor=spec_augmentor,
            is_training=True,
            fold=5,  # Use fold 5 for validation
            include_all_negatives=True  # Use all sounds as negatives for more data
        )
        
        esc50_val = ESC50Dataset(
            data_dir=esc50_path,
            preprocessor=preprocessor,
            is_training=False,
            fold=5,
            include_all_negatives=True
        )
        
        datasets_train.append(esc50_train)
        datasets_val.append(esc50_val)
        print(f"  ESC-50 train: {len(esc50_train)}, val: {len(esc50_val)}")
    
    if not datasets_train:
        raise ValueError("No training data found! Please provide data_dir or enable use_esc50")
    
    # Combine datasets
    if len(datasets_train) > 1:
        train_dataset = CombinedDataset(datasets_train)
        val_dataset = CombinedDataset(datasets_val)
    else:
        train_dataset = datasets_train[0]
        val_dataset = datasets_val[0]
    
    print(f"\nTotal training samples: {len(train_dataset)}")
    print(f"Total validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        use_weighted_sampler=True
    )
    
    # Get number of input features (stacked vertically as image height)
    # This is n_mels + n_mfcc * 2 if using MFCCs
    n_features = preprocessor.get_num_features() if hasattr(preprocessor, 'get_num_features') else config['n_mels']
    print(f"\nInput features: {n_features} (image height)")
    
    # Create model - in_channels stays 1 (single channel image)
    # The features are stacked as image height, not channels
    model = create_model(
        model_type=model_type,
        n_mels=n_features,  # Use n_features as the "height" 
        num_classes=2,
        in_channels=1  # Still single channel
    ).to(device)
    
    print(f"\nModel: {model_type}")
    print(f"Parameters: {count_parameters(model):,}")
    
    # Calculate class weights dynamically based on actual data distribution
    if hasattr(train_dataset, 'class_counts'):
        counts = train_dataset.class_counts
    else:
        # Count manually
        counts = {0: 0, 1: 0}
        for _, label in train_dataset.samples:
            counts[label] = counts.get(label, 0) + 1
    
    total = counts.get(0, 1) + counts.get(1, 1)
    weight_0 = total / (2 * max(counts.get(0, 1), 1))
    weight_1 = total / (2 * max(counts.get(1, 1), 1))
    
    # Cap the weight ratio to prevent extreme values
    max_ratio = 20.0
    if weight_1 / weight_0 > max_ratio:
        weight_1 = weight_0 * max_ratio
    
    class_weights = torch.tensor([weight_0, weight_1]).to(device)
    print(f"Class weights: non-cough={weight_0:.2f}, cough={weight_1:.2f}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience)
    
    # Resume from checkpoint
    start_epoch = 0
    best_f1 = 0.0
    
    if resume and Path(resume).exists():
        print(f"Resuming from {resume}")
        start_epoch, metrics = load_checkpoint(resume, model, optimizer)
        best_f1 = metrics.get('f1', 0.0)
        start_epoch += 1
    
    # Training loop
    print("\nStarting training...")
    
    for epoch in range(start_epoch, epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
        print(f"  Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")
        print(f"  TP: {val_metrics['tp']}, FP: {val_metrics['fp']}, FN: {val_metrics['fn']}, TN: {val_metrics['tn']}")
        
        # Save checkpoint if best F1
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                str(output_dir / 'best_model.pt'),
                config
            )
            print(f"  Saved best model (F1: {best_f1:.4f})")
        
        # Save latest checkpoint
        save_checkpoint(
            model, optimizer, epoch, val_metrics,
            str(output_dir / 'latest_model.pt'),
            config
        )
        
        # Early stopping
        if early_stopping(val_metrics['loss']):
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    print(f"\nTraining complete! Best F1: {best_f1:.4f}")
    print(f"Model saved to {output_dir / 'best_model.pt'}")
    
    return str(output_dir / 'best_model.pt')


def main():
    parser = argparse.ArgumentParser(description='Train cough detection model')
    
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory with cough/non_cough subdirectories')
    parser.add_argument('--output-dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--model-type', type=str, default='small',
                        choices=['standard', 'small', 'residual'],
                        help='Model architecture')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cpu, cuda, mps)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Data loader workers')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--no-esc50', action='store_true',
                        help='Disable ESC-50 dataset')
    parser.add_argument('--esc50-dir', type=str, default=None,
                        help='ESC-50 dataset directory')
    
    args = parser.parse_args()
    
    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=args.device,
        num_workers=args.num_workers,
        resume=args.resume,
        use_esc50=not args.no_esc50,
        esc50_dir=args.esc50_dir
    )


if __name__ == '__main__':
    main()
