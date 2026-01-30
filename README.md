# Real-Time Cough Detection ML Pipeline

A real-time machine learning pipeline that listens to your microphone and detects coughs, printing timestamps when they occur. Designed to work on Mac (both Intel and Apple Silicon), with Linux support as well.

## Quick Start

### 1. Setup Environment

**macOS / Linux:**
```bash
cd cough_detector
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

**Windows:**
```cmd
cd cough_detector
setup_windows.bat
venv\Scripts\activate
```

### 2. Install ffmpeg (required for COUGHVID dataset)

COUGHVID uses `.webm` audio files which require ffmpeg to decode.

**Windows:**
1. Download: https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip
2. Extract the zip to `C:\ffmpeg`
3. You'll have a folder like `C:\ffmpeg\ffmpeg-8.0.1-essentials_build\`
4. Add the `bin` folder to your PATH:
   - Press Windows key, type "environment variables"
   - Click "Edit the system environment variables"
   - Click "Environment Variables..."
   - Under "User variables", select "Path" and click "Edit"
   - Click "New" and add: `C:\ffmpeg\ffmpeg-8.0.1-essentials_build\bin` (adjust for your version)
   - Click OK on all windows
5. **Close and reopen** command prompt
6. Verify: `ffmpeg -version`

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt install ffmpeg
```

### 3. Train a Model

**Option A: Quick training (small dataset, ~5 min, weaker results):**
```bash
python train_quick.py
```

**Option B: Full training with COUGHVID (recommended, ~30-60 min, better results):**
```bash
pip install soundfile
python train_quick.py          # Downloads ESC-50 first
python setup_coughvid.py       # Downloads COUGHVID (~1.5GB)
python train_with_data.py      # Trains on combined dataset
```

### 3. Run Live Detection

```bash
# Start listening for coughs
python run_detection.py --model ./checkpoints/best_model.pt
```

When you cough, you'll see output like:
```
🔊 COUGH DETECTED at 2024-01-28 14:32:15.234
   Confidence: 87.3%
```

## Architecture and Model

### Data Pipeline

| Source | Coughs | Non-Coughs | Notes |
|--------|--------|------------|-------|
| COUGHVID | ~3,000 | ~1,500 | Real crowdsourced coughs, high-confidence only (>0.9) |
| ESC-50 | 40 | ~680 | Extended hard negatives (breathing, sneezing, clapping, laughing, keyboard, etc.) |
| **Total** | **~3,040** | **~2,180** | All real recordings, no synthetic data |

### Preprocessing Pipeline

```
Raw Audio File (.wav, .webm, .ogg)
       ↓
  Resample to 16kHz mono
       ↓
  Normalize amplitude to [-1, 1]
       ↓
  Pad or trim to exactly 1 second (16,000 samples)
       ↓
  Mel Spectrogram transform:
    • FFT size: 512
    • Window length: 25ms (400 samples)
    • Hop length: 10ms (160 samples)
    • Mel bands: 64
    • Frequency range: 50Hz - 8000Hz
       ↓
  Convert to decibels (log scale)
       ↓
  Normalize to [0, 1] range
       ↓
  Output: 2D tensor (1, 64, 101)
          (1 channel, 64 mel bands, 101 time frames)
```

**Why Mel Spectrograms?**
- Mimics human ear perception (logarithmic frequency scale)
- Converts audio into an "image" that CNNs can process
- Captures both frequency content and timing of coughs

### Model Architecture (CoughDetectorResidual)

```
Input: (batch, 1, 64, 101)  ← Mel spectrogram "image"
              ↓
┌─────────────────────────────────────┐
│ Conv2D(1 → 32, 7×7, stride=2)       │
│ BatchNorm2D(32)                     │
│ ReLU                                │
│ MaxPool2D(2×2)                      │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ Residual Block (32 → 64)            │
│   ├─ Conv2D(3×3) + BN + ReLU        │
│   ├─ Conv2D(3×3) + BN               │
│   └─ Skip connection (1×1 conv)     │  ← Helps gradient flow
│ ReLU                                │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ Residual Block (64 → 128)           │
│   ├─ Conv2D(3×3) + BN + ReLU        │
│   ├─ Conv2D(3×3) + BN               │
│   └─ Skip connection (1×1 conv)     │
│ ReLU                                │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ AdaptiveAvgPool2D(1×1)              │
│ Dropout(0.5)                        │
│ Linear(128 → 2)                     │
└─────────────────────────────────────┘
              ↓
Output: (batch, 2)  ← logits for [non_cough, cough]
```

**Parameters:** ~200,000

**Why Residual Architecture?**
- **Skip connections**: Prevents vanishing gradients, allows deeper networks
- **Better feature learning**: Can learn both fine and coarse patterns
- **Proven architecture**: Based on ResNet, state-of-the-art for image classification
- **Moderate size**: Good balance between capacity and speed

### Training Configuration

| Setting | Value |
|---------|-------|
| Loss | CrossEntropyLoss with dynamic class weights |
| Optimizer | AdamW (lr=0.0005, weight_decay=0.01) |
| Scheduler | Cosine annealing with warm restarts |
| Batch size | 32 |
| Max epochs | 150 |
| Early stopping | Patience of 20 epochs |
| Augmentation | Time shift, volume, gaussian noise, SpecAugment |

### Real-Time Inference Pipeline

```
Microphone Input (continuous)
         ↓
    100ms audio chunks
         ↓
┌────────────────────────────┐
│ Sliding window buffer      │
│ (1 sec window, 250ms hop)  │
└────────────────────────────┘
         ↓
    Mel spectrogram
         ↓
    CNN prediction → probability
         ↓
┌────────────────────────────┐
│ Smoothing (avg last 3)     │  ← Reduces noise
│ Threshold check (0.7)      │
│ Debounce (0.5 sec)         │  ← Prevents double-triggers
└────────────────────────────┘
         ↓
   🔊 COUGH DETECTED + timestamp
```

## Key Technical Choices

### 1. Mel Spectrogram Features
**Choice:** Mel spectrograms instead of raw waveforms or MFCCs.

**Why:**
- Mel scale mimics human auditory perception
- 2D representation enables CNN architectures proven in image classification
- Better captures the frequency characteristics of coughs (sharp onset, broadband spectrum)
- More robust than MFCCs for non-speech sounds

### 2. Depthwise Separable Convolutions
**Choice:** Used in the small model for real-time inference.

**Why:**
- Reduces parameters by ~8x compared to standard convolutions
- Maintains similar accuracy
- Critical for achieving low-latency inference on CPU

### 3. Data Augmentation Strategy
**Choice:** Both waveform and spectrogram augmentation.

**Waveform augmentations:**
- Time shifting (±20%)
- Speed perturbation (0.9x - 1.1x)
- Volume perturbation (0.7x - 1.3x)
- Gaussian noise (SNR 10-30dB)
- Background noise mixing (if noise samples available)

**Spectrogram augmentations:**
- SpecAugment: frequency masking (up to 10 bands)
- SpecAugment: time masking (up to 20 frames)

**Why:**
- Cough sounds vary significantly between people
- Need robustness to recording conditions (noise, volume, microphone quality)
- Limited labeled cough data necessitates aggressive augmentation

### 4. Sliding Window with Debouncing
**Choice:** 1-second analysis windows with 250ms hop and 0.5s debounce.

**Why:**
- 1 second captures full cough duration (typical cough: 200-500ms)
- 250ms hop provides responsive detection
- Debouncing prevents multiple triggers for a single cough event

### 5. Confidence Smoothing
**Choice:** Average predictions over 3 consecutive windows.

**Why:**
- Single-frame predictions can be noisy
- Smoothing reduces false positives without significantly increasing latency
- Coughs are sustained events, not instantaneous

## Data Sources

### Primary: COUGHVID Dataset (Recommended)
- ~25,000 crowdsourced cough recordings
- Filtered to ~1,500 high-quality samples
- Source: https://zenodo.org/records/7024894
- **License:** Creative Commons Attribution 4.0

### Secondary: ESC-50 Dataset
- 2000 environmental audio recordings
- 50 classes including "coughing" (class 24)
- 40 cough samples + similar sounds for hard negatives
- **License:** Creative Commons Attribution Non-Commercial

### Synthetic Data
- Generated cough-like sounds (noise bursts with cough envelope)
- Generated non-cough sounds (silence, hum, clicks)
- Helps with data augmentation and class balancing

**Hard negative classes used from ESC-50:**
- Breathing, snoring, sneezing (similar body sounds)
- Clapping, door knock (sharp transient sounds)
- Laughing, crying baby (vocal sounds)

## Training

### Quick Training (ESC-50 only)

```bash
python train_quick.py
```

### Full Training with Custom Data

```bash
# Prepare data (downloads ESC-50 and organizes it)
python prepare_data.py --output-dir ./data

# Train with custom data
python src/train.py \
    --data-dir ./data \
    --output-dir ./checkpoints \
    --model-type small \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.001
```

### Training Options

| Option | Default | Description |
|--------|---------|-------------|
| `--data-dir` | None | Directory with cough/non_cough subdirs |
| `--output-dir` | ./checkpoints | Where to save models |
| `--model-type` | small | Model architecture (small/standard/residual) |
| `--epochs` | 100 | Maximum training epochs |
| `--batch-size` | 32 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--patience` | 15 | Early stopping patience |
| `--device` | auto | Compute device (auto/cpu/cuda/mps) |
| `--no-esc50` | False | Disable ESC-50 dataset |

### Expected Training Results

With ESC-50 alone:
- Validation Accuracy: ~90-95%
- Validation F1 (cough class): ~0.75-0.85
- Training time: ~10-20 minutes on CPU, ~5 minutes on GPU

## Running Live Detection

### Basic Usage

```bash
python run_detection.py --model ./checkpoints/best_model.pt
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | Required | Path to trained model |
| `--threshold` | 0.7 | Detection confidence threshold (0-1) |
| `--smoothing` | 3 | Predictions to average |
| `--debounce` | 0.5 | Minimum seconds between detections |
| `--device` | auto | Compute device |
| `--audio-device` | None | Audio input device index |
| `--list-devices` | - | List available audio devices |

### Adjusting Sensitivity

**More sensitive (more detections, possibly more false positives):**
```bash
python run_detection.py --model ./checkpoints/best_model.pt --threshold 0.5 --smoothing 2
```

**Less sensitive (fewer false positives, might miss quiet coughs):**
```bash
python run_detection.py --model ./checkpoints/best_model.pt --threshold 0.85 --smoothing 4
```

### Selecting Audio Input

```bash
# List available devices
python run_detection.py --list-devices

# Use specific device
python run_detection.py --model ./checkpoints/best_model.pt --audio-device 2
```

## Project Structure

```
cough_detector/
├── src/
│   ├── __init__.py         # Package initialization
│   ├── preprocessing.py     # Audio preprocessing and mel spectrograms
│   ├── model.py            # Neural network architectures
│   ├── augmentation.py     # Data augmentation techniques
│   ├── dataset.py          # Dataset classes and data loading
│   ├── train.py            # Training loop and utilities
│   └── inference.py        # Real-time inference engine
├── checkpoints/            # Saved model checkpoints
├── datasets/               # Downloaded datasets (ESC-50)
├── data/                   # Prepared training data
├── run_detection.py        # Main entry point for live detection
├── train_quick.py          # Quick training script
├── prepare_data.py         # Data preparation utilities
├── setup.sh                # Environment setup script
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Troubleshooting

### "No audio backend available"
```bash
# macOS
brew install portaudio
pip install sounddevice

# Linux
sudo apt-get install portaudio19-dev
pip install sounddevice
```

### "Permission denied for microphone"
On macOS, grant microphone permission to Terminal/your IDE in System Preferences → Security & Privacy → Microphone.

### "CUDA out of memory"
Use CPU for inference (it's fast enough):
```bash
python run_detection.py --model ./checkpoints/best_model.pt --device cpu
```

### High false positive rate
1. Increase threshold: `--threshold 0.8`
2. Increase smoothing: `--smoothing 5`
3. Train with more diverse negative samples

### Missing cough detections
1. Decrease threshold: `--threshold 0.5`
2. Decrease debounce: `--debounce 0.3`
3. Check microphone levels - speak/cough into microphone

## License

This project uses the ESC-50 dataset which is licensed under Creative Commons Attribution Non-Commercial. The code is provided as-is for educational and research purposes.

## Acknowledgments

- ESC-50 Dataset: K. J. Piczak, "ESC: Dataset for Environmental Sound Classification"
- PyTorch and torchaudio teams
- Sounddevice library developers
