"""
Audio preprocessing pipeline for cough detection.
Converts raw audio to mel spectrograms + MFCCs + spectral features optimized for cough detection.
"""

import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from typing import Tuple, Optional


class AudioPreprocessor:
    """
    Preprocesses audio for cough detection using mel spectrograms + MFCCs + spectral features.
    
    Design choices:
    - Sample rate: 16kHz (standard for speech/medical audio, balances quality vs computation)
    - Window length: 25ms (400 samples) - captures transient cough characteristics
    - Hop length: 10ms (160 samples) - provides good temporal resolution
    - n_mels: 64 - sufficient frequency resolution for cough detection
    - n_fft: 512 - good frequency resolution at 16kHz
    - Frequency range: 100Hz-4000Hz - focused on cough frequency content (bandpass)
    - Pre-emphasis: Boosts high frequencies where cough energy is concentrated
    - MFCCs: 13 coefficients + deltas + delta-deltas for vocal characteristics
    - Spectral contrast: 7 bands to distinguish coughs from speech/noise
    - PCEN: Per-channel energy normalization for robustness to volume
    
    Total features: 64 mel + 13 mfcc + 13 delta + 13 delta-delta + 7 contrast = 110
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 64,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        f_min: float = 100.0,
        f_max: float = 4000.0,
        segment_duration: float = 1.0,
        n_mfcc: int = 13,
        use_mfcc: bool = True,
        use_pcen: bool = True,
        use_pre_emphasis: bool = True,
        pre_emphasis_coef: float = 0.97,
        use_delta_delta: bool = True,
        use_spectral_contrast: bool = True,
        n_contrast_bands: int = 6,  # Results in 7 features (6 bands + 1 valley)
        device: str = "cpu"
    ):
        """
        Initialize the audio preprocessor.
        
        Args:
            sample_rate: Target sample rate for audio
            n_mels: Number of mel filterbanks
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
            win_length: Window length for STFT
            f_min: Minimum frequency for mel filterbank (bandpass low)
            f_max: Maximum frequency for mel filterbank (bandpass high)
            segment_duration: Duration of audio segments in seconds
            n_mfcc: Number of MFCC coefficients to extract
            use_mfcc: Whether to include MFCC features
            use_pcen: Whether to use PCEN instead of log scaling
            use_pre_emphasis: Whether to apply pre-emphasis filter
            pre_emphasis_coef: Pre-emphasis coefficient (typically 0.95-0.97)
            use_delta_delta: Whether to include delta-delta (acceleration) features
            use_spectral_contrast: Whether to include spectral contrast features
            n_contrast_bands: Number of frequency bands for spectral contrast
            device: Device to use for computation
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.segment_duration = segment_duration
        self.segment_samples = int(sample_rate * segment_duration)
        self.n_mfcc = n_mfcc
        self.use_mfcc = use_mfcc
        self.use_pcen = use_pcen
        self.use_pre_emphasis = use_pre_emphasis
        self.pre_emphasis_coef = pre_emphasis_coef
        self.use_delta_delta = use_delta_delta
        self.use_spectral_contrast = use_spectral_contrast
        self.n_contrast_bands = n_contrast_bands
        self.device = device
        
        # Create mel spectrogram transform
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
            normalized=False,
            center=True,
            pad_mode="reflect"
        ).to(device)
        
        # Amplitude to dB conversion (used if not using PCEN)
        self.amplitude_to_db = T.AmplitudeToDB(
            stype="power",
            top_db=80
        ).to(device)
        
        # MFCC transform
        if use_mfcc:
            self.mfcc_transform = T.MFCC(
                sample_rate=sample_rate,
                n_mfcc=n_mfcc,
                melkwargs={
                    'n_fft': n_fft,
                    'win_length': win_length,
                    'hop_length': hop_length,
                    'n_mels': n_mels,
                    'f_min': f_min,
                    'f_max': f_max,
                }
            ).to(device)
        
        # Spectrogram transform for spectral contrast
        if use_spectral_contrast:
            self.spectrogram = T.Spectrogram(
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                power=2.0
            ).to(device)
            self.spectral_contrast = T.SpectralCentroid(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length
            ).to(device)
        
        # Resampler cache
        self._resamplers = {}
    
    def _get_resampler(self, orig_sr: int) -> T.Resample:
        """Get or create a resampler for the given sample rate."""
        if orig_sr not in self._resamplers:
            self._resamplers[orig_sr] = T.Resample(
                orig_freq=orig_sr,
                new_freq=self.sample_rate
            ).to(self.device)
        return self._resamplers[orig_sr]
    
    def load_audio(self, path: str) -> Tuple[torch.Tensor, int]:
        """
        Load an audio file and return waveform and sample rate.
        
        Args:
            path: Path to audio file
            
        Returns:
            Tuple of (waveform tensor, sample rate)
        """
        waveform, sr = torchaudio.load(path)
        return waveform, sr
    
    def resample(self, waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """
        Resample audio to target sample rate.
        
        Args:
            waveform: Input waveform tensor
            orig_sr: Original sample rate
            
        Returns:
            Resampled waveform
        """
        if orig_sr == self.sample_rate:
            return waveform
        
        resampler = self._get_resampler(orig_sr)
        return resampler(waveform)
    
    def to_mono(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert stereo audio to mono.
        
        Args:
            waveform: Input waveform tensor of shape (channels, samples)
            
        Returns:
            Mono waveform of shape (1, samples)
        """
        if waveform.shape[0] == 1:
            return waveform
        return waveform.mean(dim=0, keepdim=True)
    
    def normalize(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Normalize waveform to [-1, 1] range.
        
        Args:
            waveform: Input waveform tensor
            
        Returns:
            Normalized waveform
        """
        max_val = waveform.abs().max()
        if max_val > 0:
            return waveform / max_val
        return waveform
    
    def apply_pre_emphasis(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply pre-emphasis filter to boost high frequencies.
        
        Pre-emphasis helps with:
        - Balancing frequency spectrum (speech/coughs have more low freq energy)
        - Boosting high frequencies where cough characteristics are
        - Improving SNR for high frequencies
        
        Formula: y[n] = x[n] - coef * x[n-1]
        
        Args:
            waveform: Input waveform tensor of shape (1, samples)
            
        Returns:
            Pre-emphasized waveform
        """
        if not self.use_pre_emphasis:
            return waveform
        
        # Apply pre-emphasis: y[n] = x[n] - coef * x[n-1]
        emphasized = torch.cat([
            waveform[:, :1],  # Keep first sample
            waveform[:, 1:] - self.pre_emphasis_coef * waveform[:, :-1]
        ], dim=1)
        
        return emphasized
    
    def extract_spectral_contrast(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract spectral contrast features.
        
        Spectral contrast measures the difference between peaks and valleys
        in each frequency subband. Useful for distinguishing coughs from
        speech (coughs have sharper spectral contrast).
        
        Args:
            waveform: Input waveform tensor of shape (1, samples)
            
        Returns:
            Spectral contrast features of shape (1, n_bands+1, time_frames)
        """
        waveform = waveform.to(self.device)
        
        # Get spectrogram
        spec = self.spectrogram(waveform)  # (1, freq, time)
        
        # Calculate spectral contrast manually
        # Divide spectrum into bands and compute peak-valley difference
        n_freq = spec.shape[1]
        n_time = spec.shape[2]
        n_bands = self.n_contrast_bands
        
        # Create frequency bands (roughly logarithmic spacing)
        band_edges = torch.logspace(0, np.log10(n_freq), n_bands + 2).int()
        band_edges = torch.clamp(band_edges, 0, n_freq)
        
        contrast = torch.zeros(1, n_bands + 1, n_time, device=self.device)
        
        for i in range(n_bands):
            low = band_edges[i].item()
            high = band_edges[i + 1].item()
            if high <= low:
                high = low + 1
            if high > n_freq:
                high = n_freq
            
            band = spec[:, low:high, :]
            if band.shape[1] > 0:
                # Peak (top 20%) minus valley (bottom 20%)
                sorted_band, _ = torch.sort(band, dim=1)
                n_bins = sorted_band.shape[1]
                top_idx = max(1, int(n_bins * 0.8))
                bot_idx = max(1, int(n_bins * 0.2))
                
                peaks = sorted_band[:, top_idx:, :].mean(dim=1)
                valleys = sorted_band[:, :bot_idx, :].mean(dim=1)
                
                contrast[:, i, :] = torch.log1p(peaks) - torch.log1p(valleys)
        
        # Add spectral centroid as last feature
        centroid = self.spectral_contrast(waveform)  # (1, time)
        # Normalize centroid
        centroid = centroid / (self.sample_rate / 2)  # Normalize to [0, 1]
        contrast[:, -1, :centroid.shape[1]] = centroid
        
        # Normalize contrast features
        contrast = (contrast - contrast.mean()) / (contrast.std() + 1e-8)
        
        return contrast
    
    def apply_pcen(self, mel_spec: torch.Tensor, 
                   alpha: float = 0.98,
                   delta: float = 2.0, 
                   r: float = 0.5,
                   eps: float = 1e-6) -> torch.Tensor:
        """
        Apply Per-Channel Energy Normalization (PCEN).
        
        PCEN is more robust to volume variations than log scaling.
        
        Args:
            mel_spec: Mel spectrogram (power)
            alpha: Smoothing coefficient
            delta: Bias
            r: Compression exponent
            eps: Small constant for numerical stability
            
        Returns:
            PCEN-normalized spectrogram
        """
        # Smooth the spectrogram over time (simple IIR filter approximation)
        # For simplicity, use a moving average
        smooth = torch.nn.functional.avg_pool2d(
            mel_spec.unsqueeze(0), 
            kernel_size=(1, 10), 
            stride=(1, 1), 
            padding=(0, 5)
        ).squeeze(0)
        
        # Trim to match original size
        smooth = smooth[:, :, :mel_spec.shape[2]]
        
        # PCEN formula: (mel_spec / (eps + smooth)^alpha + delta)^r - delta^r
        pcen = (mel_spec / (eps + smooth).pow(alpha) + delta).pow(r) - delta ** r
        
        return pcen
    
    def compute_deltas(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute delta (first derivative) features.
        
        Args:
            features: Feature tensor of shape (channels, freq, time)
            
        Returns:
            Delta features of same shape
        """
        # Pad for edge handling
        padded = torch.nn.functional.pad(features, (1, 1), mode='replicate')
        # Simple difference
        deltas = (padded[:, :, 2:] - padded[:, :, :-2]) / 2
        return deltas
    
    def pad_or_trim(self, waveform: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        """
        Pad or trim waveform to target length.
        
        Args:
            waveform: Input waveform tensor of shape (1, samples)
            length: Target length in samples (defaults to segment_samples)
            
        Returns:
            Waveform of shape (1, length)
        """
        if length is None:
            length = self.segment_samples
            
        current_length = waveform.shape[1]
        
        if current_length == length:
            return waveform
        elif current_length > length:
            # Trim from center for better cough capture
            start = (current_length - length) // 2
            return waveform[:, start:start + length]
        else:
            # Pad with zeros
            padding = length - current_length
            pad_left = padding // 2
            pad_right = padding - pad_left
            return torch.nn.functional.pad(waveform, (pad_left, pad_right), mode='constant', value=0)
    
    def extract_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract mel spectrogram from waveform.
        
        Args:
            waveform: Input waveform tensor of shape (1, samples)
            
        Returns:
            Mel spectrogram of shape (1, n_mels, time_frames)
        """
        waveform = waveform.to(self.device)
        mel_spec = self.mel_spectrogram(waveform)
        
        if self.use_pcen:
            # Apply PCEN normalization
            mel_spec = self.apply_pcen(mel_spec)
            # Normalize to [0, 1] range
            mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-8)
        else:
            # Standard log scaling
            mel_spec_db = self.amplitude_to_db(mel_spec)
            # Normalize to [0, 1] range for neural network input
            mel_spec = (mel_spec_db + 80) / 80  # Since top_db=80
            mel_spec = mel_spec.clamp(0, 1)
        
        return mel_spec
    
    def extract_mfcc(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract MFCC features from waveform.
        
        Args:
            waveform: Input waveform tensor of shape (1, samples)
            
        Returns:
            MFCCs of shape (1, n_mfcc, time_frames)
        """
        waveform = waveform.to(self.device)
        mfcc = self.mfcc_transform(waveform)
        
        # Normalize MFCCs
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
        
        return mfcc
    
    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract all features (mel spectrogram + MFCCs + delta-delta + spectral contrast).
        
        Args:
            waveform: Input waveform tensor of shape (1, samples)
            
        Returns:
            Combined features tensor of shape (1, n_features, time_frames)
            
        Features (when all enabled):
            - 64 mel spectrogram bands
            - 13 MFCCs
            - 13 MFCC deltas
            - 13 MFCC delta-deltas
            - 7 spectral contrast bands
            = 110 total features
        """
        # Apply pre-emphasis before feature extraction
        waveform_emph = self.apply_pre_emphasis(waveform)
        
        # Extract mel spectrogram (from pre-emphasized signal)
        mel_spec = self.extract_mel_spectrogram(waveform_emph)
        
        feature_list = [mel_spec]
        min_time = mel_spec.shape[2]
        
        if self.use_mfcc:
            # Extract MFCCs (from pre-emphasized signal)
            mfcc = self.extract_mfcc(waveform_emph)
            
            # Compute delta MFCCs (first derivative)
            mfcc_delta = self.compute_deltas(mfcc)
            
            feature_list.append(mfcc)
            feature_list.append(mfcc_delta)
            min_time = min(min_time, mfcc.shape[2], mfcc_delta.shape[2])
            
            # Compute delta-delta MFCCs (second derivative / acceleration)
            if self.use_delta_delta:
                mfcc_delta_delta = self.compute_deltas(mfcc_delta)
                feature_list.append(mfcc_delta_delta)
                min_time = min(min_time, mfcc_delta_delta.shape[2])
        
        if self.use_spectral_contrast:
            # Extract spectral contrast (from original signal for cleaner contrast)
            contrast = self.extract_spectral_contrast(waveform)
            feature_list.append(contrast)
            min_time = min(min_time, contrast.shape[2])
        
        # Ensure all features have same time dimension
        feature_list = [f[:, :, :min_time] for f in feature_list]
        
        # Stack features vertically (along frequency axis)
        # Result: (1, total_features, time)
        features = torch.cat(feature_list, dim=1)
        
        return features
    
    def process(self, waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """
        Full preprocessing pipeline: resample -> mono -> normalize -> pad/trim -> features.
        
        Args:
            waveform: Input waveform tensor
            orig_sr: Original sample rate
            
        Returns:
            Features ready for model input
        """
        # Resample to target sample rate
        waveform = self.resample(waveform, orig_sr)
        
        # Convert to mono
        waveform = self.to_mono(waveform)
        
        # Normalize
        waveform = self.normalize(waveform)
        
        # Pad or trim to fixed length
        waveform = self.pad_or_trim(waveform)
        
        # Extract features (mel spec + MFCCs if enabled)
        features = self.extract_features(waveform)
        
        return features
    
    def process_file(self, path: str) -> torch.Tensor:
        """
        Load and preprocess an audio file.
        
        Args:
            path: Path to audio file
            
        Returns:
            Features ready for model input
        """
        waveform, sr = self.load_audio(path)
        return self.process(waveform, sr)
    
    def get_expected_time_frames(self) -> int:
        """Calculate expected number of time frames in output spectrogram."""
        return (self.segment_samples // self.hop_length) + 1
    
    def get_num_features(self) -> int:
        """Get the number of feature channels (for model input size)."""
        n_features = self.n_mels
        
        if self.use_mfcc:
            n_features += self.n_mfcc  # MFCCs
            n_features += self.n_mfcc  # MFCC deltas
            
            if self.use_delta_delta:
                n_features += self.n_mfcc  # MFCC delta-deltas
        
        if self.use_spectral_contrast:
            n_features += self.n_contrast_bands + 1  # Contrast bands + centroid
        
        return n_features


class RealtimePreprocessor(AudioPreprocessor):
    """
    Preprocessor optimized for real-time streaming audio.
    Uses a sliding window approach with configurable overlap.
    """
    
    def __init__(
        self,
        window_duration: float = 1.0,
        hop_duration: float = 0.5,
        **kwargs
    ):
        """
        Initialize real-time preprocessor.
        
        Args:
            window_duration: Duration of analysis window in seconds
            hop_duration: Hop between windows in seconds
            **kwargs: Additional arguments for AudioPreprocessor
        """
        super().__init__(segment_duration=window_duration, **kwargs)
        self.window_duration = window_duration
        self.hop_duration = hop_duration
        self.window_samples = int(self.sample_rate * window_duration)
        self.hop_samples = int(self.sample_rate * hop_duration)
        
        # Buffer for streaming audio
        self.buffer = torch.zeros(1, 0)
    
    def add_audio(self, audio_chunk: torch.Tensor) -> list:
        """
        Add audio chunk to buffer and return any complete windows.
        
        Args:
            audio_chunk: Audio chunk tensor of shape (samples,) or (1, samples)
            
        Returns:
            List of feature tensors for complete windows
        """
        # Ensure correct shape
        if audio_chunk.dim() == 1:
            audio_chunk = audio_chunk.unsqueeze(0)
        
        # Append to buffer
        self.buffer = torch.cat([self.buffer, audio_chunk], dim=1)
        
        # Extract complete windows
        features_list = []
        while self.buffer.shape[1] >= self.window_samples:
            window = self.buffer[:, :self.window_samples]
            
            # Process window
            window = self.normalize(window)
            features = self.extract_features(window)
            features_list.append(features)
            
            # Advance buffer
            self.buffer = self.buffer[:, self.hop_samples:]
        
        return features_list
    
    def reset(self):
        """Reset the audio buffer."""
        self.buffer = torch.zeros(1, 0)


def create_preprocessor(realtime: bool = False, **kwargs) -> AudioPreprocessor:
    """
    Factory function to create appropriate preprocessor.
    
    Args:
        realtime: Whether to create a real-time preprocessor
        **kwargs: Additional preprocessor arguments
        
    Returns:
        AudioPreprocessor or RealtimePreprocessor instance
    """
    if realtime:
        return RealtimePreprocessor(**kwargs)
    return AudioPreprocessor(**kwargs)
