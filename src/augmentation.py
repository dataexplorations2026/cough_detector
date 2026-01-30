"""
Data augmentation techniques for audio cough detection.

Augmentation is critical for cough detection because:
1. Limited labeled cough data availability
2. Need robustness to different recording environments
3. Need generalization across different people's coughs
"""

import torch
import torchaudio
import torchaudio.transforms as T
import random
import numpy as np
from typing import Tuple, Optional, List
from pathlib import Path


class AudioAugmentor:
    """
    Collection of audio augmentation techniques.
    
    Augmentations applied in waveform domain:
    - Time shifting
    - Speed perturbation
    - Adding background noise
    - Volume/gain perturbation
    
    Augmentations applied in spectrogram domain:
    - SpecAugment (time and frequency masking)
    - Time stretching (via spectrogram)
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        noise_dir: Optional[str] = None,
        p_augment: float = 0.5
    ):
        """
        Initialize augmentor.
        
        Args:
            sample_rate: Target sample rate
            noise_dir: Directory containing background noise files
            p_augment: Probability of applying each augmentation
        """
        self.sample_rate = sample_rate
        self.p_augment = p_augment
        
        # Load background noises if provided
        self.noise_samples = []
        if noise_dir and Path(noise_dir).exists():
            self._load_noise_samples(noise_dir)
    
    def _load_noise_samples(self, noise_dir: str, max_samples: int = 100):
        """Load background noise samples from directory."""
        noise_path = Path(noise_dir)
        audio_extensions = ['.wav', '.mp3', '.flac', '.ogg']
        
        noise_files = []
        for ext in audio_extensions:
            noise_files.extend(noise_path.glob(f'*{ext}'))
        
        for f in noise_files[:max_samples]:
            try:
                waveform, sr = torchaudio.load(str(f))
                if sr != self.sample_rate:
                    resampler = T.Resample(sr, self.sample_rate)
                    waveform = resampler(waveform)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                self.noise_samples.append(waveform)
            except Exception:
                continue
    
    def time_shift(
        self,
        waveform: torch.Tensor,
        shift_limit: float = 0.2
    ) -> torch.Tensor:
        """
        Randomly shift audio in time.
        
        Args:
            waveform: Input waveform (1, samples)
            shift_limit: Maximum shift as fraction of total length
            
        Returns:
            Time-shifted waveform
        """
        if random.random() > self.p_augment:
            return waveform
        
        shift_amount = int(waveform.shape[1] * random.uniform(-shift_limit, shift_limit))
        
        if shift_amount > 0:
            # Shift right (pad left, trim right)
            waveform = torch.nn.functional.pad(waveform, (shift_amount, 0))[:, :-shift_amount]
        elif shift_amount < 0:
            # Shift left (pad right, trim left)
            shift_amount = abs(shift_amount)
            waveform = torch.nn.functional.pad(waveform, (0, shift_amount))[:, shift_amount:]
        
        return waveform
    
    def speed_perturbation(
        self,
        waveform: torch.Tensor,
        speed_range: Tuple[float, float] = (0.9, 1.1)
    ) -> torch.Tensor:
        """
        Apply speed perturbation (DISABLED - causes memory issues on Windows).
        Just returns the original waveform.
        """
        # Disabled due to memory allocation issues with resampling
        return waveform
    
    def add_noise(
        self,
        waveform: torch.Tensor,
        snr_range: Tuple[float, float] = (5, 20)
    ) -> torch.Tensor:
        """
        Add background noise at random SNR.
        
        Args:
            waveform: Input waveform
            snr_range: Range of signal-to-noise ratios in dB
            
        Returns:
            Noisy waveform
        """
        if random.random() > self.p_augment or len(self.noise_samples) == 0:
            return waveform
        
        # Select random noise sample
        noise = random.choice(self.noise_samples).clone()
        
        # Match length
        target_len = waveform.shape[1]
        noise_len = noise.shape[1]
        
        if noise_len < target_len:
            # Repeat noise
            repeats = (target_len // noise_len) + 1
            noise = noise.repeat(1, repeats)
        
        # Random crop
        start = random.randint(0, noise.shape[1] - target_len)
        noise = noise[:, start:start + target_len]
        
        # Calculate SNR and mix
        snr_db = random.uniform(*snr_range)
        signal_power = waveform.pow(2).mean()
        noise_power = noise.pow(2).mean()
        
        if noise_power > 0:
            snr_linear = 10 ** (snr_db / 10)
            noise_scale = torch.sqrt(signal_power / (snr_linear * noise_power))
            waveform = waveform + noise_scale * noise
        
        return waveform
    
    def add_gaussian_noise(
        self,
        waveform: torch.Tensor,
        snr_range: Tuple[float, float] = (10, 30)
    ) -> torch.Tensor:
        """
        Add Gaussian noise at random SNR.
        
        Args:
            waveform: Input waveform
            snr_range: Range of signal-to-noise ratios in dB
            
        Returns:
            Noisy waveform
        """
        if random.random() > self.p_augment:
            return waveform
        
        snr_db = random.uniform(*snr_range)
        signal_power = waveform.pow(2).mean()
        
        noise = torch.randn_like(waveform)
        noise_power = noise.pow(2).mean()
        
        snr_linear = 10 ** (snr_db / 10)
        noise_scale = torch.sqrt(signal_power / (snr_linear * noise_power))
        
        return waveform + noise_scale * noise
    
    def volume_perturbation(
        self,
        waveform: torch.Tensor,
        gain_range: Tuple[float, float] = (0.7, 1.3)
    ) -> torch.Tensor:
        """
        Apply random volume change.
        
        Args:
            waveform: Input waveform
            gain_range: Range of gain factors
            
        Returns:
            Volume-perturbed waveform
        """
        if random.random() > self.p_augment:
            return waveform
        
        gain = random.uniform(*gain_range)
        return waveform * gain
    
    def pitch_shift(
        self,
        waveform: torch.Tensor,
        shift_range: Tuple[int, int] = (-2, 2)
    ) -> torch.Tensor:
        """
        Apply pitch shift in semitones.
        
        Args:
            waveform: Input waveform
            shift_range: Range of pitch shift in semitones
            
        Returns:
            Pitch-shifted waveform
        """
        if random.random() > self.p_augment:
            return waveform
        
        n_steps = random.randint(*shift_range)
        if n_steps == 0:
            return waveform
        
        # Use torchaudio's pitch shift
        try:
            effects = [["pitch", str(n_steps * 100)]]  # cents
            waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform, self.sample_rate, effects
            )
        except Exception:
            # Fallback: no pitch shift if sox not available
            pass
        
        return waveform
    
    def augment(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply random combination of augmentations.
        
        Args:
            waveform: Input waveform
            
        Returns:
            Augmented waveform
        """
        # Apply augmentations in sequence (each has probability p_augment)
        waveform = self.time_shift(waveform)
        waveform = self.speed_perturbation(waveform)
        waveform = self.volume_perturbation(waveform)
        waveform = self.add_gaussian_noise(waveform)
        
        if len(self.noise_samples) > 0:
            waveform = self.add_noise(waveform)
        
        return waveform


class SpecAugment:
    """
    SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition
    https://arxiv.org/abs/1904.08779
    
    Applies time and frequency masking to spectrograms.
    """
    
    def __init__(
        self,
        freq_mask_param: int = 10,
        time_mask_param: int = 20,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
        p: float = 0.5
    ):
        """
        Initialize SpecAugment.
        
        Args:
            freq_mask_param: Maximum frequency mask width
            time_mask_param: Maximum time mask width
            n_freq_masks: Number of frequency masks to apply
            n_time_masks: Number of time masks to apply
            p: Probability of applying augmentation
        """
        self.freq_mask = T.FrequencyMasking(freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param)
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.p = p
    
    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to spectrogram.
        
        Args:
            spectrogram: Input spectrogram (batch, channels, freq, time) or (channels, freq, time)
            
        Returns:
            Augmented spectrogram
        """
        if random.random() > self.p:
            return spectrogram
        
        # Handle batch dimension
        squeeze = False
        if spectrogram.dim() == 3:
            spectrogram = spectrogram.unsqueeze(0)
            squeeze = True
        
        for _ in range(self.n_freq_masks):
            spectrogram = self.freq_mask(spectrogram)
        
        for _ in range(self.n_time_masks):
            spectrogram = self.time_mask(spectrogram)
        
        if squeeze:
            spectrogram = spectrogram.squeeze(0)
        
        return spectrogram


class MixUp:
    """
    MixUp augmentation for audio classification.
    Mixes two samples with a random weight.
    """
    
    def __init__(self, alpha: float = 0.2):
        """
        Initialize MixUp.
        
        Args:
            alpha: Beta distribution parameter
        """
        self.alpha = alpha
    
    def __call__(
        self,
        x1: torch.Tensor,
        y1: torch.Tensor,
        x2: torch.Tensor,
        y2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply MixUp to a pair of samples.
        
        Args:
            x1, x2: Input features
            y1, y2: One-hot encoded labels
            
        Returns:
            Mixed features and labels
        """
        lam = np.random.beta(self.alpha, self.alpha)
        x = lam * x1 + (1 - lam) * x2
        y = lam * y1 + (1 - lam) * y2
        return x, y


def create_augmentation_pipeline(
    sample_rate: int = 16000,
    noise_dir: Optional[str] = None,
    p_augment: float = 0.5,
    use_spec_augment: bool = True
) -> Tuple[AudioAugmentor, Optional[SpecAugment]]:
    """
    Create augmentation pipeline.
    
    Args:
        sample_rate: Audio sample rate
        noise_dir: Directory with noise samples
        p_augment: Augmentation probability
        use_spec_augment: Whether to use SpecAugment
        
    Returns:
        Tuple of (audio augmentor, spec augment)
    """
    audio_aug = AudioAugmentor(
        sample_rate=sample_rate,
        noise_dir=noise_dir,
        p_augment=p_augment
    )
    
    spec_aug = SpecAugment(p=p_augment) if use_spec_augment else None
    
    return audio_aug, spec_aug
