"""
Real-time cough detection inference engine.
Listens to microphone and detects coughs in real-time.
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import Optional, Callable
import threading
import queue

# Audio input handling
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import RealtimePreprocessor
from src.model import create_model


class CoughDetectorInference:
    """
    Real-time cough detection inference engine.
    
    Features:
    - Sliding window analysis
    - Confidence smoothing to reduce false positives
    - Debouncing to avoid multiple detections for same cough
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = 'auto',
        confidence_threshold: float = 0.5,
        smoothing_window: int = 3,
        debounce_seconds: float = 0.5,
        verbose: bool = True
    ):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on ('auto', 'cpu', 'cuda', 'mps')
            confidence_threshold: Minimum confidence for detection
            smoothing_window: Number of predictions to average
            debounce_seconds: Minimum time between detections
            verbose: Print debug information
        """
        self.verbose = verbose
        self.confidence_threshold = confidence_threshold
        self.debounce_seconds = debounce_seconds
        
        # Setup device
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = torch.device(device)
        if self.verbose:
            print(f"Using device: {self.device}")
        
        # Load model and config
        self._load_model(model_path)
        
        # Create preprocessor with config from checkpoint
        self.preprocessor = RealtimePreprocessor(
            sample_rate=self.config.get('sample_rate', 16000),
            n_mels=self.config.get('n_mels', 64),
            n_fft=self.config.get('n_fft', 512),
            hop_length=self.config.get('hop_length', 160),
            win_length=self.config.get('win_length', 400),
            f_min=self.config.get('f_min', 100.0),
            f_max=self.config.get('f_max', 4000.0),
            window_duration=self.config.get('segment_duration', 1.0),
            hop_duration=0.25,  # 250ms hop for real-time
            n_mfcc=self.config.get('n_mfcc', 13),
            use_mfcc=self.config.get('use_mfcc', True),
            use_pcen=self.config.get('use_pcen', True),
            use_pre_emphasis=self.config.get('use_pre_emphasis', True),
            pre_emphasis_coef=self.config.get('pre_emphasis_coef', 0.97),
            use_delta_delta=self.config.get('use_delta_delta', True),
            use_spectral_contrast=self.config.get('use_spectral_contrast', True),
            n_contrast_bands=self.config.get('n_contrast_bands', 6),
            device='cpu'
        )
        
        # Prediction smoothing
        self.prediction_history = deque(maxlen=smoothing_window)
        
        # Debouncing
        self.last_detection_time = 0
        
        # Callbacks
        self.on_cough_detected: Optional[Callable[[datetime, float], None]] = None
    
    def _load_model(self, model_path: str):
        """Load trained model from checkpoint."""
        if self.verbose:
            print(f"Loading model from {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        self.config = checkpoint.get('config', {})
        
        # Calculate number of features (image height)
        n_mels = self.config.get('n_mels', 64)
        n_mfcc = self.config.get('n_mfcc', 13)
        use_mfcc = self.config.get('use_mfcc', True)
        use_delta_delta = self.config.get('use_delta_delta', True)
        use_spectral_contrast = self.config.get('use_spectral_contrast', True)
        n_contrast_bands = self.config.get('n_contrast_bands', 6)
        
        n_features = n_mels
        if use_mfcc:
            n_features += n_mfcc  # MFCCs
            n_features += n_mfcc  # MFCC deltas
            if use_delta_delta:
                n_features += n_mfcc  # MFCC delta-deltas
        if use_spectral_contrast:
            n_features += n_contrast_bands + 1  # Contrast bands + centroid
        
        # Create model - in_channels stays 1, features are image height
        model_type = self.config.get('model_type', 'small')
        self.model = create_model(
            model_type=model_type,
            n_mels=n_features,  # Use n_features as image height
            num_classes=2,
            in_channels=1  # Single channel
        )
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        if self.verbose:
            print(f"Model loaded: {model_type}")
            metrics = checkpoint.get('metrics', {})
            if metrics:
                print(f"  Validation F1: {metrics.get('f1', 'N/A'):.4f}")
    
    @torch.no_grad()
    def predict(self, spectrogram: torch.Tensor) -> tuple:
        """
        Make prediction on a spectrogram.
        
        Args:
            spectrogram: Feature tensor (1, n_features, time_frames) or (batch, 1, n_features, time_frames)
            
        Returns:
            Tuple of (is_cough, confidence)
        """
        spectrogram = spectrogram.to(self.device)
        
        # Ensure batch dimension: (1, features, time) -> (1, 1, features, time)
        if spectrogram.dim() == 3:
            spectrogram = spectrogram.unsqueeze(0)
        
        # Forward pass
        logits = self.model(spectrogram)
        probs = torch.softmax(logits, dim=1)
        
        # Cough probability (class 1)
        cough_prob = probs[0, 1].item()
        
        return cough_prob > 0.5, cough_prob
    
    def process_audio_chunk(self, audio_chunk: np.ndarray) -> Optional[tuple]:
        """
        Process an audio chunk and return detection result.
        
        Args:
            audio_chunk: Audio samples as numpy array
            
        Returns:
            Tuple of (timestamp, confidence) if cough detected, None otherwise
        """
        # Convert to tensor
        if isinstance(audio_chunk, np.ndarray):
            audio_chunk = torch.from_numpy(audio_chunk.astype(np.float32))
        
        # Ensure 1D or 2D (channels, samples)
        if audio_chunk.dim() == 1:
            audio_chunk = audio_chunk.unsqueeze(0)
        
        # Convert to mono if stereo
        if audio_chunk.shape[0] > 1:
            audio_chunk = audio_chunk.mean(dim=0, keepdim=True)
        
        # Add to preprocessor buffer and get spectrograms
        spectrograms = self.preprocessor.add_audio(audio_chunk)
        
        for spec in spectrograms:
            is_cough, confidence = self.predict(spec)
            
            # Add to history for smoothing
            self.prediction_history.append(confidence)
            
            # Calculate smoothed confidence
            smoothed_confidence = np.mean(self.prediction_history)
            
            # Check threshold and debounce
            current_time = datetime.now().timestamp()
            time_since_last = current_time - self.last_detection_time
            
            if (smoothed_confidence >= self.confidence_threshold and 
                time_since_last >= self.debounce_seconds):
                
                self.last_detection_time = current_time
                timestamp = datetime.now()
                
                # Call callback if set
                if self.on_cough_detected:
                    self.on_cough_detected(timestamp, smoothed_confidence)
                
                return timestamp, smoothed_confidence
        
        return None
    
    def reset(self):
        """Reset internal state."""
        self.preprocessor.reset()
        self.prediction_history.clear()
        self.last_detection_time = 0


class RealtimeMicrophoneDetector:
    """
    Real-time microphone listener for cough detection.
    """
    
    def __init__(
        self,
        inference_engine: CoughDetectorInference,
        sample_rate: int = 16000,
        chunk_duration: float = 0.1,
        device_index: Optional[int] = None,
        backend: str = 'auto'
    ):
        """
        Initialize microphone detector.
        
        Args:
            inference_engine: Inference engine instance
            sample_rate: Audio sample rate
            chunk_duration: Duration of each audio chunk in seconds
            device_index: Audio input device index (None for default)
            backend: Audio backend ('auto', 'sounddevice', 'pyaudio')
        """
        self.inference = inference_engine
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration)
        self.device_index = device_index
        
        # Select backend
        if backend == 'auto':
            if SOUNDDEVICE_AVAILABLE:
                backend = 'sounddevice'
            elif PYAUDIO_AVAILABLE:
                backend = 'pyaudio'
            else:
                raise RuntimeError("No audio backend available. Install sounddevice or pyaudio.")
        
        self.backend = backend
        
        # State
        self.running = False
        self.audio_queue = queue.Queue()
        
        # Detection callbacks
        self.on_detection: Optional[Callable[[datetime, float], None]] = None
    
    def _audio_callback_sounddevice(self, indata, frames, time, status):
        """Callback for sounddevice."""
        if status:
            print(f"Audio status: {status}")
        self.audio_queue.put(indata.copy())
    
    def _process_audio(self):
        """Process audio from queue."""
        while self.running:
            try:
                audio_chunk = self.audio_queue.get(timeout=0.5)
                audio_chunk = audio_chunk.flatten()
                
                result = self.inference.process_audio_chunk(audio_chunk)
                
                if result is not None:
                    timestamp, confidence = result
                    
                    # Print detection
                    print(f"\n🔊 COUGH DETECTED at {timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
                    print(f"   Confidence: {confidence:.2%}")
                    
                    if self.on_detection:
                        self.on_detection(timestamp, confidence)
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing audio: {e}")
    
    def start(self):
        """Start listening to microphone."""
        if self.running:
            return
        
        self.running = True
        self.inference.reset()
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self._process_audio)
        self.process_thread.start()
        
        if self.backend == 'sounddevice':
            self._start_sounddevice()
        elif self.backend == 'pyaudio':
            self._start_pyaudio()
    
    def _start_sounddevice(self):
        """Start audio capture with sounddevice."""
        print(f"\nStarting microphone capture (sounddevice)...")
        print(f"Sample rate: {self.sample_rate} Hz")
        print(f"Chunk size: {self.chunk_size} samples ({self.chunk_size/self.sample_rate*1000:.0f}ms)")
        
        # List available devices
        if self.inference.verbose:
            print("\nAvailable audio devices:")
            print(sd.query_devices())
            print()
        
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            blocksize=self.chunk_size,
            device=self.device_index,
            callback=self._audio_callback_sounddevice
        )
        self.stream.start()
        
        print("Listening for coughs... (Press Ctrl+C to stop)")
    
    def _start_pyaudio(self):
        """Start audio capture with pyaudio."""
        print(f"\nStarting microphone capture (pyaudio)...")
        
        self.pa = pyaudio.PyAudio()
        
        # List devices
        if self.inference.verbose:
            print("\nAvailable audio devices:")
            for i in range(self.pa.get_device_count()):
                info = self.pa.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    print(f"  [{i}] {info['name']}")
            print()
        
        def callback(in_data, frame_count, time_info, status):
            audio = np.frombuffer(in_data, dtype=np.float32)
            self.audio_queue.put(audio)
            return (None, pyaudio.paContinue)
        
        self.stream = self.pa.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk_size,
            stream_callback=callback
        )
        self.stream.start_stream()
        
        print("Listening for coughs... (Press Ctrl+C to stop)")
    
    def stop(self):
        """Stop listening."""
        self.running = False
        
        if hasattr(self, 'stream'):
            if self.backend == 'sounddevice':
                self.stream.stop()
                self.stream.close()
            elif self.backend == 'pyaudio':
                self.stream.stop_stream()
                self.stream.close()
                self.pa.terminate()
        
        if hasattr(self, 'process_thread'):
            self.process_thread.join(timeout=2.0)
        
        print("\nStopped listening.")
    
    def run(self):
        """Run the detector (blocking)."""
        self.start()
        
        try:
            while self.running:
                import time
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.stop()


def list_audio_devices():
    """List available audio input devices."""
    print("Available audio input devices:\n")
    
    if SOUNDDEVICE_AVAILABLE:
        print("sounddevice devices:")
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                print(f"  [{i}] {dev['name']} ({dev['max_input_channels']} ch)")
        print()
    
    if PYAUDIO_AVAILABLE:
        print("pyaudio devices:")
        pa = pyaudio.PyAudio()
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"  [{i}] {info['name']} ({info['maxInputChannels']} ch)")
        pa.terminate()


def main():
    parser = argparse.ArgumentParser(description='Real-time cough detection')
    
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='Confidence threshold for detection (0-1)')
    parser.add_argument('--smoothing', type=int, default=3,
                        help='Number of predictions to average')
    parser.add_argument('--debounce', type=float, default=0.5,
                        help='Minimum seconds between detections')
    parser.add_argument('--device', type=str, default='auto',
                        help='Compute device (auto, cpu, cuda, mps)')
    parser.add_argument('--audio-device', type=int, default=None,
                        help='Audio input device index')
    parser.add_argument('--backend', type=str, default='auto',
                        choices=['auto', 'sounddevice', 'pyaudio'],
                        help='Audio backend')
    parser.add_argument('--list-devices', action='store_true',
                        help='List available audio devices and exit')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # List devices and exit
    if args.list_devices:
        list_audio_devices()
        return
    
    # Create inference engine
    inference = CoughDetectorInference(
        model_path=args.model,
        device=args.device,
        confidence_threshold=args.threshold,
        smoothing_window=args.smoothing,
        debounce_seconds=args.debounce,
        verbose=not args.quiet
    )
    
    # Create microphone detector
    detector = RealtimeMicrophoneDetector(
        inference_engine=inference,
        sample_rate=inference.config.get('sample_rate', 16000),
        device_index=args.audio_device,
        backend=args.backend
    )
    
    # Run
    detector.run()


if __name__ == '__main__':
    main()
