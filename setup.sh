#!/bin/bash
# Setup script for cough detector
# Creates a virtual environment and installs all dependencies

set -e

echo "=== Cough Detector Setup ==="
echo ""

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macOS"
    # Check architecture
    if [[ $(uname -m) == "arm64" ]]; then
        ARCH="Apple Silicon (M1/M2/M3)"
    else
        ARCH="Intel"
    fi
    echo "Detected: $OS ($ARCH)"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="Linux"
    echo "Detected: $OS"
else
    OS="Other"
    echo "Detected: $OS (may require manual dependency installation)"
fi

echo ""

# Check Python version
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Python not found. Please install Python 3.9 or higher."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
echo "Python version: $PYTHON_VERSION"

# Create virtual environment
VENV_DIR="venv"
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists."
    read -p "Do you want to recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
        echo "Creating new virtual environment..."
        $PYTHON_CMD -m venv "$VENV_DIR"
    fi
else
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv "$VENV_DIR"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with appropriate backend
echo ""
echo "Installing PyTorch..."

if [[ "$OS" == "macOS" ]]; then
    # macOS: Use MPS (Metal Performance Shaders) for Apple Silicon
    pip install torch torchaudio
elif [[ "$OS" == "Linux" ]]; then
    # Linux: Try CUDA first, fallback to CPU
    if command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA GPU detected, installing CUDA version..."
        pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        echo "No NVIDIA GPU detected, installing CPU version..."
        pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
else
    pip install torch torchaudio
fi

# Install other dependencies
echo ""
echo "Installing other dependencies..."
pip install sounddevice numpy pandas scikit-learn tqdm

# macOS specific: Install PortAudio for sounddevice
if [[ "$OS" == "macOS" ]]; then
    echo ""
    echo "Checking for PortAudio (required for audio input)..."
    if command -v brew &> /dev/null; then
        if ! brew list portaudio &> /dev/null; then
            echo "Installing PortAudio via Homebrew..."
            brew install portaudio
        else
            echo "PortAudio already installed."
        fi
    else
        echo "Note: Homebrew not found. If audio input doesn't work, install PortAudio manually:"
        echo "  brew install portaudio"
    fi
fi

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import torch; import torchaudio; import sounddevice; print('All dependencies installed successfully!')"

# Check for available audio devices
echo ""
echo "Checking audio input devices..."
python -c "import sounddevice as sd; print(sd.query_devices())"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To train a model:"
echo "  python src/train.py --output-dir ./checkpoints"
echo ""
echo "To run live detection:"
echo "  python run_detection.py --model ./checkpoints/best_model.pt"
echo ""
