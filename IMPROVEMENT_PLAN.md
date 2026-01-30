# Cough Detector: Current Issues and Improvement Plan

## Overview

This document describes the current state of the cough detection system, known issues, and a plan to achieve production-quality performance.

**Current status:** The model is non-functional and requires retraining with properly curated data.

---

## Critical Issues

### Issue 1: Training Infrastructure Was Broken

The `torchaudio.load()` function requires the `torchcodec` package, which was not installed. This caused every audio file load to fail silently, and the dataset returned zero tensors as a fallback.

**Result:** The saved model was trained entirely on zeros and learned nothing about audio. It outputs >99% cough probability for all inputs, including silence.

**Evidence from saved model:**
```
Accuracy: 36.98% (worse than random guessing)
Recall: 1.0 (predicts "cough" for everything)
True Negatives: 0 (never predicted "not cough")
```

**Status:** 
- ✅ `torchcodec` is now installed
- ✅ Dataset code now crashes on load failure instead of silently returning zeros
- ⚠️ Model must be retrained

### Issue 2: Training Data Has Severe Quality Problems

Manual inspection of the COUGHVID-derived training data reveals multiple critical issues:

#### A. Mislabeled Samples

Some files in `data/non_cough/` actually contain coughs:
- `coughvid_other_0008.wav` - contains cough
- `coughvid_other_0010.wav` - contains cough  
- `coughvid_other_0021.wav` - contains cough
- (likely many more)

> **Note on file naming:** File names are assigned by a loop counter during data preparation (`coughvid_other_{i:04d}.wav`). The ordering depends on pandas DataFrame iteration order, which should be deterministic for the same dataset version but may vary if the source data or preparation script changes.

#### B. Non-Cough Class is Mostly Silence

The majority of `data/non_cough/` samples are silence or near-silence, not meaningful negative examples. The model needs to learn what sounds are NOT coughs, not just that silence isn't a cough.

#### C. Insufficient Foreground Speech

While some speech exists in the non-cough data, it is typically:
- Background speech (distant, muffled)
- Not representative of close-mic speech that the detector will encounter

The model needs foreground speech samples to learn that normal talking ≠ cough.

#### D. Duration Mismatch

| Aspect | Training Data | Real-Time Detection |
|--------|---------------|---------------------|
| Typical duration | ~10 seconds | 1-2 second windows |
| Cough duration | 1-2 seconds | 1-2 seconds |

**Problem:** The audio files are ~10 seconds long, but coughs only last 1-2 seconds. This creates two issues:

1. **For real-time detection:** Using 10-second sliding windows would be impractically slow
2. **For naive slicing:** If you slice 10-second "cough" files into 1-2 second chunks, approximately 80-90% of resulting chunks would be silence (the parts before/after the actual cough), severely polluting the cough class with mislabeled silence

---

## Current Data Composition

| Source | Cough Files | Non-Cough Files | Notes |
|--------|-------------|-----------------|-------|
| COUGHVID | ~3,000 | ~5,000 | 10-sec recordings, many quality issues |
| ESC-50 | 40 | ~180 | 5-sec environmental sounds |
| **Total** | **~3,040** | **~5,180** | **Requires curation before use** |

### What's Missing from Non-Cough Class

| Sound Type | Current Status | Importance |
|------------|---------------|------------|
| Foreground speech | ❌ Missing | Critical |
| Throat clearing | ❌ Missing | High |
| Laughing | ✅ Some (ESC-50) | High |
| Sneezing | ✅ Some (ESC-50) | Medium |
| Breathing | ✅ Some (ESC-50) | Medium |
| Background noise | ⚠️ Overrepresented (silence) | Low |

---

## Model and Features Assessment

The model architecture and feature extraction are appropriate for the task:

| Component | Implementation | Assessment |
|-----------|---------------|------------|
| Features | Mel spectrogram (64) + MFCCs (13) + deltas (13) | ✅ Industry standard |
| Model | ResNet-style CNN, ~200K parameters | ✅ Appropriate |
| Frequency range | 100-4000 Hz | ✅ Covers cough frequencies |
| Sample rate | 16 kHz | ✅ Standard |
| Window size | 1 second | ✅ Appropriate for coughs |

**The model and features are not the problem. Data quality is the problem.**

---

## Improvement Plan

### Phase 1: Fix Training Data (Required)

#### Step 1.1: Curate Cough Samples

**Goal:** Extract clean 1-2 second cough segments from the 10-second recordings.

**Approach:**
```python
def extract_cough_segments(input_dir, output_dir):
    """
    For each cough recording:
    1. Detect high-energy segments (likely coughs)
    2. Extract 1-2 second windows around energy peaks
    3. Verify segment contains actual cough (manual spot-check)
    4. Save as individual files
    """
```

**Considerations:**
- Use energy-based detection to find cough onset
- Include ~200ms padding before/after the cough
- Discard segments that are pure silence
- Target: 3,000-5,000 clean 1-2 second cough segments

#### Step 1.2: Build Quality Non-Cough Dataset

**Goal:** Replace the current non-cough data with meaningful negative examples.

**Required categories:**

| Category | Source | Target Count |
|----------|--------|--------------|
| Foreground speech | LibriSpeech, Common Voice, or self-recorded | 2,000-3,000 |
| Throat clearing | Self-recorded | 100-200 |
| Laughing | ESC-50 + self-recorded | 200-300 |
| Breathing/sighing | ESC-50 + self-recorded | 200-300 |
| Sneezing | ESC-50 | 100-200 |
| Background noise | Various | 500-1,000 |
| Silence | Minimal | 100-200 |

**Speech data options:**

1. **LibriSpeech** (recommended)
   - Source: https://www.openslr.org/12/
   - Download `dev-clean` subset (~350 MB)
   - Segment into 1-2 second clips
   - License: CC BY 4.0

2. **Common Voice**
   - Source: https://commonvoice.mozilla.org/
   - More varied recording quality (closer to real-world)
   - License: CC0

3. **Self-recorded**
   - Fastest for testing
   - Record 10-20 minutes of normal speech
   - Segment into 1-2 second clips

#### Step 1.3: Remove Mislabeled Samples

**Approach:**
```python
def audit_non_cough_samples(data_dir, sample_size=200):
    """
    1. Randomly sample N files from non_cough/
    2. Play each and mark as: 'correct', 'mislabeled', 'unclear'
    3. Move mislabeled files to cough/ or delete
    4. Report mislabeling rate to estimate total contamination
    """
```

If mislabeling rate is >5%, consider discarding all COUGHVID-derived non-cough samples and rebuilding from scratch.

### Phase 2: Retrain Model

After data curation:

```bash
cd /Users/jon/git/cough_detector
source venv/bin/activate
python train_with_data.py
```

**Expected training metrics (with good data):**
- Accuracy: >85%
- Cough recall: >90%
- Cough precision: >80%
- F1 score: >0.85

### Phase 3: Validate and Tune

#### Validation Tests

1. **Silence test:** Model should NOT fire on ambient room noise
2. **Speech test:** Model should NOT fire on normal conversation
3. **Cough test:** Model SHOULD detect actual coughs
4. **Threshold tuning:** Adjust `--threshold` based on precision/recall tradeoff

#### Expected Behavior After Fix

| Input | Expected Output |
|-------|-----------------|
| Silence | No detection |
| Normal speech | No detection |
| Loud speech | No detection |
| Throat clearing | No detection (or rare) |
| Actual cough | Detection with >70% confidence |

---

## Data Preparation Scripts Needed

### Script 1: Cough Segment Extractor

```python
# extract_cough_segments.py

import torch
import torchaudio
from pathlib import Path

def find_energy_peaks(waveform, sr, threshold_db=-30, min_duration=0.1):
    """Find high-energy segments that likely contain coughs."""
    # Compute short-time energy
    # Find segments above threshold
    # Return list of (start_time, end_time) tuples
    pass

def extract_segments(input_dir: Path, output_dir: Path, segment_duration: float = 1.0):
    """Extract cough segments from long recordings."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for audio_file in input_dir.glob("*.wav"):
        waveform, sr = torchaudio.load(audio_file)
        
        # Find cough locations
        peaks = find_energy_peaks(waveform, sr)
        
        for i, (start, end) in enumerate(peaks):
            # Extract segment with padding
            # Save to output_dir
            pass
```

### Script 2: Speech Data Preparer

```python
# prepare_speech_data.py

def prepare_librispeech(librispeech_dir: Path, output_dir: Path, max_samples: int = 3000):
    """
    Process LibriSpeech data:
    1. Find all .flac files
    2. Segment into 1-2 second clips
    3. Convert to 16kHz mono WAV
    4. Save to output_dir
    """
    pass
```

### Script 3: Data Auditor

```python
# audit_data.py

def audit_dataset(data_dir: Path):
    """
    Interactive tool to audit random samples:
    - Play audio
    - Mark as correct/mislabeled
    - Generate report
    """
    pass
```

---

## Quick Start After Data Fixes

Once data is curated:

```bash
# 1. Verify data
python -c "
from pathlib import Path
cough = len(list(Path('data/cough').glob('*.wav')))
non_cough = len(list(Path('data/non_cough').glob('*.wav')))
print(f'Coughs: {cough}, Non-coughs: {non_cough}')
print(f'Ratio: 1:{non_cough/cough:.1f}')
"

# 2. Train
python train_with_data.py

# 3. Test on silence (should NOT fire)
python run_detection.py --model checkpoints/best_model.pt --threshold 0.7

# 4. Test on speech (should NOT fire)  
# Play speech near microphone

# 5. Test on coughs (SHOULD fire)
# Cough near microphone
```

---

## Success Criteria

| Metric | Target | How to Measure |
|--------|--------|----------------|
| No false positives on silence | 0 detections/minute | Run detector in quiet room |
| No false positives on speech | <1 detection/minute | Talk normally near mic |
| Cough detection rate | >80% | Cough 10 times, count detections |
| Detection latency | < 2s | Time from cough to detection print |

---

## Files Modified

| File | Change |
|------|--------|
| `src/dataset.py` | Now crashes on audio load failure instead of returning zeros |
| `requirements.txt` | Should include `torchcodec` |

---

## Summary

The cough detector's poor performance stems from two issues:

1. **Infrastructure bug (fixed):** Audio loading was silently failing, causing the model to train on zeros
2. **Data quality (not yet fixed):** Training data has mislabeled samples, is mostly silence, lacks speech, and has duration mismatches

**Next action:** Curate training data by extracting clean cough segments and building a proper non-cough dataset with foreground speech.
