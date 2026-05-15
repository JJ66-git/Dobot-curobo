#!/bin/bash
set -e

export PYTHONNOUSERSITE=1

echo "=== Dobot X-Trainer motion planning workstation setup ==="

echo "[1/6] Check GPU..."
nvidia-smi || { echo "Error: NVIDIA GPU not found"; exit 1; }

echo "[2/6] Create conda environment..."
conda create -n curobo python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate curobo

echo "[3/6] Install PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo "[4/6] Install cuRobo..."
cd ~
if [ ! -d "curobo" ]; then
    git clone https://github.com/NVlabs/curobo.git
fi
cd curobo
pip install -e .

echo "[5/6] Clone project..."
cd ~
if [ ! -d "Dobot-curobo" ]; then
    git clone https://github.com/JJ66-git/Dobot-curobo.git
fi
cd Dobot-curobo

echo "[6/6] Verify project-local robot config..."
test -f "$(pwd)/curobo/my_x_trainer/xtrainer.yml"
test -f "$(pwd)/curobo/my_x_trainer/xtrainer.urdf"

pip install numpy scipy pyyaml

echo ""
echo "=== Setup complete ==="
echo ""
echo "Notes:"
echo "  Motion-planning code now resolves curobo/my_x_trainer using project-local absolute paths."
echo "  No xtrainer symlink is written into the cuRobo installation directory."
echo ""
echo "Next:"
echo "  conda activate curobo"
echo "  cd ~/Dobot-curobo/x-trainer/source/leisaac"
echo "  python -m leisaac.motion_planning.test_motion_planning"
