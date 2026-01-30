@echo off
echo ============================================================
echo Cough Detector - Full Setup and Training
echo ============================================================
echo.

REM Check if we're in the right directory
if not exist "src\train.py" (
    echo ERROR: Please run this from the cough_detector folder
    echo Example: cd C:\cough_detector\cough_detector
    exit /b 1
)

REM Step 1: Setup virtual environment
echo [1/7] Setting up virtual environment...
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate
    
    echo Upgrading pip...
    python -m pip install --upgrade pip -q
    
    echo Installing PyTorch...
    nvidia-smi >nul 2>&1
    if errorlevel 1 (
        echo No NVIDIA GPU detected, installing CPU version...
        pip install torch torchaudio -q
    ) else (
        echo NVIDIA GPU detected, installing CUDA version...
        pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
    )
    
    echo Installing other dependencies...
    pip install sounddevice numpy pandas scikit-learn tqdm -q
) else (
    echo Virtual environment already exists, activating...
    call venv\Scripts\activate
)

REM Step 2: Install soundfile
echo.
echo [2/7] Installing soundfile...
pip install soundfile -q

REM Step 3: Clean old data (fresh start)
echo.
echo [3/7] Cleaning old data for fresh start...
if exist "data" rmdir /s /q data
if exist "checkpoints" rmdir /s /q checkpoints
if exist "datasets" rmdir /s /q datasets

REM Step 4: Download ESC-50 (without training)
echo.
echo [4/7] Downloading ESC-50 dataset...
python download_esc50.py

REM Step 5: Download and prepare COUGHVID
echo.
echo [5/7] Downloading and preparing COUGHVID dataset (this may take 10-20 min)...
python setup_coughvid.py

REM Step 6: Train the model
echo.
echo [6/7] Training the model (this will take 30-60 min)...
python train_with_data.py

REM Step 7: Done!
echo.
echo [7/7] Setup complete!
echo.
echo ============================================================
echo DONE! To run cough detection:
echo.
echo   venv\Scripts\activate
echo   python run_detection.py --model checkpoints\best_model.pt --threshold 0.7 --smoothing 1
echo.
echo ============================================================
