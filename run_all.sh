#!/bin/bash
echo "============================================================"
echo "Cough Detector - Full Setup and Training"
echo "============================================================"
echo

# Check if we're in the right directory
if [ ! -f "src/train.py" ]; then
    echo "ERROR: Please run this from the cough_detector folder"
    echo "Example: cd cough_detector"
    exit 1
fi

# Step 1: Setup virtual environment
echo "[1/7] Setting up virtual environment..."
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    
    echo "Upgrading pip..."
    pip install --upgrade pip -q
    
    echo "Installing PyTorch..."
    pip install torch torchaudio -q
    
    echo "Installing other dependencies..."
    pip install sounddevice numpy pandas scikit-learn tqdm -q
else
    echo "Virtual environment already exists, activating..."
    source venv/bin/activate
fi

# Step 2: Install soundfile
echo
echo "[2/7] Installing soundfile..."
pip install soundfile -q

# Step 3: Clean old data (fresh start)
echo
echo "[3/7] Cleaning old data for fresh start..."
rm -rf data checkpoints datasets

# Step 4: Download ESC-50 (without training)
echo
echo "[4/7] Downloading ESC-50 dataset..."
python download_esc50.py

# Step 5: Download and prepare COUGHVID
echo
echo "[5/7] Downloading and preparing COUGHVID dataset (this may take 10-20 min)..."
python setup_coughvid.py

# Step 6: Train the model
echo
echo "[6/7] Training the model (this will take 30-60 min)..."
python train_with_data.py

# Step 7: Done!
echo
echo "[7/7] Setup complete!"
echo
echo "============================================================"
echo "DONE! To run cough detection:"
echo
echo "  source venv/bin/activate"
echo "  python run_detection.py --model checkpoints/best_model.pt --threshold 0.7 --smoothing 1"
echo
echo "============================================================"
