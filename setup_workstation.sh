#!/bin/bash
set -e

echo "=== Dobot X-Trainer 运动规划模块 - 工作站部署脚本 ==="

# 1. 检查 GPU
echo "[1/6] 检查 GPU..."
nvidia-smi || { echo "错误: 未检测到 NVIDIA GPU"; exit 1; }

# 2. 创建 conda 环境
echo "[2/6] 创建 conda 环境..."
conda create -n curobo python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate curobo

# 3. 安装 PyTorch
echo "[3/6] 安装 PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. 安装 cuRobo
echo "[4/6] 安装 cuRobo..."
cd ~
if [ ! -d "curobo" ]; then
    git clone https://github.com/NVlabs/curobo.git
fi
cd curobo
pip install -e .

# 5. 克隆项目
echo "[5/6] 克隆项目..."
cd ~
if [ ! -d "Dobot-curobo" ]; then
    git clone git@github.com:JJ66-git/Dobot-curobo.git
fi
cd Dobot-curobo

# 6. 配置符号链接
echo "[6/6] 配置符号链接..."
mkdir -p ~/curobo/src/curobo/content/configs/robot
ln -sf ~/Dobot-curobo/curobo/my_x_trainer ~/curobo/src/curobo/content/configs/robot/xtrainer

# 安装基础依赖
pip install numpy scipy pyyaml

echo ""
echo "=== 部署完成！==="
echo ""
echo "下一步："
echo "  conda activate curobo"
echo "  cd ~/Dobot-curobo/x-trainer/source/leisaac"
echo "  python -m leisaac.motion_planning.test_motion_planning"
