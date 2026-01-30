@echo off
REM Setup script for cough detector (Windows)
REM Creates a virtual environment and installs all dependencies

echo === Cough Detector Setup (Windows) ===
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.9 or higher.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Python found:
python --version

REM Create virtual environment
if exist venv (
    echo Virtual environment already exists.
    set /p RECREATE="Do you want to recreate it? (y/n): "
    if /i "%RECREATE%"=="y" (
        rmdir /s /q venv
        echo Creating new virtual environment...
        python -m venv venv
    )
) else (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install PyTorch
echo.
echo Installing PyTorch...
REM Check for NVIDIA GPU
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo No NVIDIA GPU detected, installing CPU version...
    pip install torch torchaudio
) else (
    echo NVIDIA GPU detected, installing CUDA version...
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
)

REM Install other dependencies
echo.
echo Installing other dependencies...
pip install sounddevice numpy pandas scikit-learn tqdm

REM Verify installation
echo.
echo Verifying installation...
python -c "import torch; import torchaudio; import sounddevice; print('All dependencies installed successfully!')"

REM Check audio devices
echo.
echo Checking audio input devices...
python -c "import sounddevice as sd; print(sd.query_devices())"

echo.
echo === Setup Complete ===
echo.
echo To activate the environment:
echo   venv\Scripts\activate
echo.
echo IMPORTANT: For COUGHVID dataset (.webm files), you need ffmpeg:
echo   1. Download from https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip
echo   2. Extract and add the bin folder to your PATH
echo   3. Restart command prompt
echo.
echo To train a model:
echo   python train_quick.py
echo.
echo To train with COUGHVID (better quality):
echo   python setup_coughvid.py
echo   python train_with_data.py
echo.
echo To run live detection:
echo   python run_detection.py --model checkpoints\best_model.pt
echo.
pause
