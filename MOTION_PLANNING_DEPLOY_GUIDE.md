# Dobot X-Trainer 运动规划模块 — 工作站部署与调试指南

> 本文档指导你将 Windows 上开发的运动规划模块迁移到 Linux GPU 工作站，
> 完成环境配置、单元测试、集成调试的全流程。

---

## 目录

1. [环境要求](#1-环境要求)
2. [克隆项目到工作站](#2-克隆项目到工作站)
3. [安装 Python 依赖](#3-安装-python-依赖)
4. [配置 cuRobo](#4-配置-curobo)
5. [验证 URDF 模型](#5-验证-urdf-模型)
6. [运行运动规划模块测试](#6-运行运动规划模块测试)
7. [与 Isaac Sim 集成](#7-与-isaac-sim-集成)
8. [常见问题排查](#8-常见问题排查)
9. [调试技巧](#9-调试技巧)

---

## 1. 环境要求

### 硬件

| 项目 | 最低要求 | 推荐配置 |
|------|---------|---------|
| GPU | NVIDIA RTX 2060 (6GB) | RTX 3080+ (10GB+) |
| 内存 | 16 GB | 32 GB |
| 磁盘 | 20 GB 可用空间 | 50 GB（含 Isaac Sim） |
| CPU | 4 核 | 8 核+ |

### 软件

| 软件 | 版本 | 说明 |
|------|------|------|
| 操作系统 | Ubuntu 20.04 / 22.04 | 推荐 22.04 |
| NVIDIA 驱动 | >= 525.x | `nvidia-smi` 检查 |
| CUDA | >= 11.8 | cuRobo 依赖 |
| Python | 3.10 | Isaac Sim 自带或 conda 环境 |
| PyTorch | >= 2.0 (CUDA) | `torch.cuda.is_available()` 必须为 True |
| Isaac Sim | 2023.1+ | 仿真环境（可选，测试阶段可不用） |
| cuRobo | 0.8.0+ | GPU 运动规划核心库 |

### 检查命令速查

```bash
# GPU 驱动
nvidia-smi

# CUDA 版本
nvcc --version

# Python 版本
python3 --version

# PyTorch + CUDA
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

---

## 2. 克隆项目到工作站

### 2.1 SSH 方式（推荐）

```bash
# 如果还没配置 SSH key，先生成
ssh-keygen -t ed25519 -C "your_email@example.com"
cat ~/.ssh/id_ed25519.pub
# 把输出的公钥添加到 GitHub: Settings → SSH and GPG keys → New SSH key

# 克隆
cd ~
git clone git@github.com:JJ66-git/Dobot-curobo.git
cd Dobot-curobo
```

### 2.2 HTTPS 方式

```bash
cd ~
git clone https://github.com/JJ66-git/Dobot-curobo.git
cd Dobot-curobo
```

### 2.3 验证项目结构

```bash
tree -L 3 -I __pycache__
```

你应该看到类似这样的结构：

```
Dobot-curobo/
├── .gitignore
├── MOTION_PLANNING_TUTORIAL.md
├── MOTION_PLANNING_DEPLOY_GUIDE.md    ← 本文件
├── curobo/
│   └── my_x_trainer/
│       ├── build_model.py
│       ├── xtrainer.urdf
│       └── xtrainer.yml
└── x-trainer/
    └── source/leisaac/leisaac/
        └── motion_planning/
            ├── __init__.py
            ├── coordinate_transform.py
            ├── ik_solver.py
            ├── motion_planner.py
            ├── planner_interface.py
            ├── task_state_machine.py
            └── test_motion_planning.py
```

---

## 3. 安装 Python 依赖

### 3.1 创建 conda 环境（推荐）

```bash
# 创建独立环境
conda create -n curobo python=3.10 -y
conda activate curobo

# 安装 PyTorch（根据你的 CUDA 版本选择）
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 验证
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

> **输出应为：** `PyTorch: 2.x.x` 和 `CUDA: True`

### 3.2 安装基础依赖

```bash
pip install numpy scipy pyyaml
```

### 3.3 安装 Isaac Sim（如果需要仿真集成）

```bash
# 方法 A：通过 pip 安装（推荐）
pip install isaacsim-rl isaacsim-replicator isaacsim-extscache-physics isaacsim-extscache-kit-sdk isaacsim-app

# 方法 B：通过 NVIDIA Omniverse Launcher 安装
# 从 https://developer.nvidia.com/isaac-sim 下载
# 安装后设置环境变量：
# export ISAACSIM_PATH=~/.local/share/ov/pkg/isaac-sim-2023.1.1
# source $ISAACSIM_PATH/setup_conda_env.sh
```

---

## 4. 配置 cuRobo

### 4.1 安装 cuRobo

```bash
# 克隆 cuRobo
cd ~
git clone https://github.com/NVlabs/curobo.git
cd curobo

# 安装（会编译 CUDA 内核，需要几分钟）
pip install -e .

# 验证安装
python -c "from curobo.wrap.reacher.ik_solver import IKSolver; print('cuRobo OK')"
```

> **注意：** 首次安装 cuRobo 会编译 CUDA kernel，可能需要 5-10 分钟。编译失败通常是因为 CUDA 版本不匹配。

### 4.2 配置机器人模型路径

cuRobo 需要知道你的 URDF 和 YAML 配置文件在哪里。有两种方式：

**方式 A：符号链接（推荐）**

```bash
# 将你的机器人配置链接到 cuRobo 的资源目录
ln -s ~/Dobot-curobo/curobo/my_x_trainer ~/curobo/src/curobo/content/configs/robot/xtrainer
```

**方式 B：环境变量**

```bash
# 在 ~/.bashrc 或 conda 环境变量中添加
export CUROBO_CONFIG_DIR=~/Dobot-curobo/curobo/my_x_trainer

# 使生效
source ~/.bashrc
```

### 4.3 验证 cuRobo 能找到机器人配置

```python
# test_curobo_config.py
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

try:
    config = IKSolverConfig.from_robot_yaml(
        "xtrainer.yml",  # cuRobo 会在配置路径中查找
        num_seeds=4,
        self_collision_check=True,
    )
    solver = IKSolver(config)
    print("[OK] cuRobo 成功加载 xtrainer.yml")
except Exception as e:
    print(f"[FAIL] 加载失败: {e}")
```

运行：
```bash
python test_curobo_config.py
```

---

## 5. 验证 URDF 模型

### 5.1 用 cuRobo 验证 URDF

```python
# test_urdf_validation.py
import torch
from curobo.geom.types import Cuboid
from curobo.types.math import Pose
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

# 加载 IK 求解器
config = IKSolverConfig.from_robot_yaml(
    "xtrainer.yml",
    num_seeds=32,
    self_collision_check=True,
)
solver = IKSolver(config)

# 测试：求解一个简单的目标位姿
target = Pose(
    position=torch.tensor([[0.3, 0.2, 0.3]]).cuda(),
    quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).cuda(),
)

result = solver.solve_single(target)

print(f"成功: {result.success.item()}")
print(f"关节角度: {result.joint_position.cpu().numpy().tolist()}")
print(f"位置误差: {result.position_error.item() * 1000:.2f} mm")
print(f"姿态误差: {result.rotation_error.item():.4f} rad")
```

运行：
```bash
python test_urdf_validation.py
```

### 5.2 用 build_model.py 重新生成配置（如果修改了 URDF）

```bash
cd ~/Dobot-curobo/curobo/my_x_trainer
python build_model.py
# 会重新生成 xtrainer.yml
```

---

## 6. 运行运动规划模块测试

### 6.1 运行完整测试套件

```bash
cd ~/Dobot-curobo

# 运行测试
python -m x-trainer.source.leisaac.leisaac.motion_planning.test_motion_planning
```

或者直接运行：

```bash
cd ~/Dobot-curobo/x-trainer/source/leisaac
python -m leisaac.motion_planning.test_motion_planning
```

### 6.2 预期输出

测试套件包含 5 个测试，输出类似：

```
============================================================
  Motion Planning Module - End-to-End Tests
============================================================

[TEST 1] Coordinate Transform
--------------------------------------------------
  [OK] pixel_to_camera: z=0.5m
  [OK] camera_to_base: valid 4x4 matrix
  [OK] full pipeline: depth=0.5m
  [OK] invalid depth: rejected
  [OK] arm fk: right_ee_link
--------------------------------------------------
Result: 5/5 passed

[TEST 2] IK Solver
--------------------------------------------------
  [OK] left arm IK: success
  [OK] right arm IK: success
  [OK] both arms IK: both solved
  [OK] check_ik_valid: True
  [OK] forward_kinematics: valid position
--------------------------------------------------
Result: 5/5 passed

[TEST 3] Motion Planner
--------------------------------------------------
  [OK] plan_to_pose: 120 points, 2.40s
  [OK] plan_joint_to_joint: trajectory valid
  [OK] plan_grasp: 3-phase complete
  [OK] update_obstacles: 2 obstacles
--------------------------------------------------
Result: 4/4 passed

[TEST 4] State Machine
--------------------------------------------------
  [OK] state transitions: full cycle
  [OK] error handling: ERROR state
  [OK] waypoint manager: 3 waypoints
  [OK] dual arm coordinator: left+right
--------------------------------------------------
Result: 4/4 passed

[TEST 5] Full Pipeline
--------------------------------------------------
  [OK] warmup: 2.15s
  [OK] transform: valid 4x4
  [OK] plan: PlanResult(success=True)
  [OK] to_json: valid JSON
  [OK] grasp_sequence: grasp+place
--------------------------------------------------
Result: 5/5 passed

============================================================
  ALL PASS  23/23 tests passed
============================================================
```

### 6.3 单独运行某个测试（调试用）

如果某个测试失败，可以单独运行它来排查：

```python
# 在 Python 中运行单个测试
from leisaac.motion_planning.test_motion_planning import (
    test_coordinate_transform,
    test_ik_solver,
    test_motion_planner,
    test_state_machine,
    test_full_pipeline,
)

# 只运行 IK 测试
test_ik_solver()
```

---

## 7. 与 Isaac Sim 集成

### 7.1 启动 Isaac Sim 并加载场景

```python
# 在 Isaac Sim 的 Python 环境中运行
from omni.isaac.core import World
from leisaac.motion_planning import MotionPlanningModule

# 创建仿真世界
world = World(stage_units_in_meters=1.0)

# 加载你的场景（桌子、篮子等）
# ... 这里是你的场景加载代码 ...

# 初始化运动规划模块
module = MotionPlanningModule()
module.warmup()  # 首次调用会初始化 cuRobo，耗时几秒

# 获取当前关节状态（从仿真中读取）
# module.set_joint_state("left", [0, 0, 0, 0, 0, 0])
# module.set_joint_state("right", [0, 0, 0, 0, 0, 0])
```

### 7.2 完整的抓取流程示例

```python
from leisaac.motion_planning import MotionPlanningModule, SceneBuilder

module = MotionPlanningModule()
module.warmup()

# 1. 设置障碍物（桌子、篮子等）
scene = SceneBuilder()
scene.add_table(
    name="work_table",
    position=[0.5, 0.0, 0.0],
    dimensions=[0.8, 1.0, 0.02],
)
scene.add_basket(
    name="basket",
    position=[0.5, 0.3, 0.15],
    dimensions=[0.25, 0.2, 0.15],
)

# 2. 从相机获取目标物体位姿（相机坐标系）
# pixel_u, pixel_v = 320, 240  # 检测到的像素坐标
# depth = 0.5  # 深度值（米）

# 3. 转换到基座坐标系
# base_pose = module.transform_camera_to_base(
#     pixel_u, pixel_v, depth,
#     camera_intrinsics=your_camera_intrinsics,
#     camera_extrinsics=your_camera_extrinsics,
# )

# 4. 规划抓取
# result = module.plan_grasp_sequence(
#     target_pose=base_pose,
#     place_pose={"position": [0.5, 0.3, 0.25], "quaternion": [1,0,0,0]},
#     obstacles=scene.get_obstacles(),
#     arm="left",
# )

# if result.success:
#     trajectory = result.trajectory
#     # 将轨迹发送给仿真中的机器人执行
#     for waypoint in trajectory:
#         # 设置关节位置
#         pass
```

### 7.3 在 policy_server 中集成

如果你使用 gRPC policy_server 架构：

```python
# 在 policy_server.py 中添加
from leisaac.motion_planning import MotionPlanningModule

class PolicyServer:
    def __init__(self):
        self.planner = MotionPlanningModule()
        self.planner.warmup()

    def PlanGrasp(self, request, context):
        """gRPC 接口：规划抓取动作"""
        target_pose = {
            "position": [request.x, request.y, request.z],
            "quaternion": [request.qw, request.qx, request.qy, request.qz],
        }
        obstacles = self._parse_obstacles(request.obstacles)

        result = self.planner.plan(target_pose, obstacles, arm=request.arm)

        return PlanResponse(
            success=result.success,
            trajectory=result.trajectory,
            duration=result.duration,
            error_message=result.error_message,
        )
```

---

## 8. 常见问题排查

### 8.1 `CUDA not available` 或 `torch.cuda.is_available()` 返回 False

```bash
# 检查驱动
nvidia-smi

# 检查 PyTorch CUDA 版本是否匹配
python -c "import torch; print(torch.version.cuda)"

# 如果不匹配，重装 PyTorch
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 8.2 cuRobo 安装失败（编译 CUDA kernel 出错）

```bash
# 确认 CUDA toolkit 版本
nvcc --version

# 确认 gcc 版本（cuRobo 需要 gcc >= 7）
gcc --version

# 如果 gcc 太旧
sudo apt install gcc-11 g++-11
export CC=gcc-11
export CXX=g++-11

# 重新安装 cuRobo
cd ~/curobo
pip install -e . --no-cache-dir
```

### 8.3 `ModuleNotFoundError: No module named 'leisaac'`

```bash
# 确保你在正确的目录下运行
cd ~/Dobot-curobo/x-trainer/source/leisaac
python -m leisaac.motion_planning.test_motion_planning

# 或者把路径加到 PYTHONPATH
export PYTHONPATH=$PYTHONPATH:~/Dobot-curobo/x-trainer/source/leisaac
python -c "from leisaac.motion_planning import MotionPlanningModule; print('OK')"
```

### 8.4 IK 求解全部失败（success=False）

可能原因：
1. **目标超出工作空间** — 检查目标位置是否在机器人臂长范围内（约 0.5m）
2. **URDF 关节限位过窄** — 检查 `xtrainer.urdf` 中的 `limit` 标签
3. **cuRobo 配置错误** — 重新运行 `build_model.py` 生成配置

```python
# 快速诊断：测试一个肯定在工作空间内的目标
target = {"position": [0.2, 0.1, 0.2], "quaternion": [1, 0, 0, 0]}
result = module.plan(target, obstacles=[], arm="left")
print(result)
```

### 8.5 运动规划失败（plan_to_pose 返回 success=False）

可能原因：
1. **碰撞** — 目标位姿或路径上有障碍物，用 `SceneBuilder` 检查场景
2. **关节限位** — 路径上某些关节超出限位
3. **规划超时** — 增加 `max_time` 参数（默认 2 秒）

```python
# 增加规划时间
planner = DualArmMotionPlanner(max_time=5.0)
```

### 8.6 `FileNotFoundError: xtrainer.yml not found`

```bash
# 确认文件存在
ls ~/Dobot-curobo/curobo/my_x_trainer/xtrainer.yml

# 确认 cuRobo 能找到它
# 方式 A：符号链接
ln -s ~/Dobot-curobo/curobo/my_x_trainer ~/curobo/src/curobo/content/configs/robot/xtrainer

# 方式 B：在代码中使用绝对路径
config = IKSolverConfig.from_robot_yaml(
    "/home/your_username/Dobot-curobo/curobo/my_x_trainer/xtrainer.yml",
    ...
)
```

---

## 9. 调试技巧

### 9.1 用 cuRobo 可视化工具检查机器人

```python
# 在 Isaac Sim 中可视化机器人
from omni.isaac.core.robots import Robot

# 加载 URDF
robot = Robot(
    prim_path="/World/XTrainer",
    usd_path="~/Dobot-curobo/curobo/my_x_trainer/xtrainer.urdf",
    name="xtrainer",
)
world.scene.add(robot)
world.reset()

# 设置关节位置，观察机器人姿态
robot.set_joint_positions([0.5, -0.3, 0.2, -0.1, 0.4, -0.2] * 2)
```

### 9.2 打印 IK 求解过程

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 或者只看 cuRobo 的日志
logging.getLogger("curobo").setLevel(logging.DEBUG)
```

### 9.3 导出轨迹为 CSV 分析

```python
import csv

result = module.plan(target_pose, obstacles=[], arm="left")
if result.success:
    with open("trajectory.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "j1", "j2", "j3", "j4", "j5", "j6"])
        for i, q in enumerate(result.trajectory):
            t = i * 0.02  # dt=0.02s
            writer.writerow([t] + q)
    print("轨迹已保存到 trajectory.csv")
```

### 9.4 用 matplotlib 可视化关节轨迹

```python
import matplotlib.pyplot as plt
import csv

times, joints = [], []
with open("trajectory.csv") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        times.append(float(row[0]))
        joints.append([float(x) for x in row[1:]])

joints = list(zip(*joints))  # 转置

fig, axes = plt.subplots(3, 2, figsize=(12, 8))
for i, ax in enumerate(axes.flat):
    ax.plot(times, joints[i])
    ax.set_title(f"Joint {i+1}")
    ax.set_ylabel("Angle (rad)")
    ax.set_xlabel("Time (s)")
    ax.grid(True)
plt.tight_layout()
plt.savefig("joint_trajectory.png", dpi=150)
print("图已保存到 joint_trajectory.png")
```

### 9.5 性能分析

```python
import time

# 测量 IK 求解时间
target = {"position": [0.3, 0.2, 0.3], "quaternion": [1, 0, 0, 0]}

# 预热
module.warmup()

# 计时
times = []
for _ in range(100):
    start = time.perf_counter()
    result = module.plan(target, obstacles=[], arm="left")
    times.append(time.perf_counter() - start)

print(f"IK 求解: 平均 {sum(times)/len(times)*1000:.1f} ms")
print(f"IK 求解: 最快 {min(times)*1000:.1f} ms")
print(f"IK 求解: 最慢 {max(times)*1000:.1f} ms")
```

---

## 附录 A：快速部署脚本

将以下内容保存为 `setup_workstation.sh`，一键部署：

```bash
#!/bin/bash
set -e

echo "=== Dobot X-Trainer 运动规划模块 - 工作站部署脚本 ==="

# 1. 检查 GPU
echo "[1/6] 检查 GPU..."
nvidia-smi || { echo "错误: 未检测到 NVIDIA GPU"; exit 1; }

# 2. 创建 conda 环境
echo "[2/6] 创建 conda 环境..."
conda create -n curobo python=3.10 -y
conda activate curobo

# 3. 安装 PyTorch
echo "[3/6] 安装 PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. 安装 cuRobo
echo "[4/6] 安装 cuRobo..."
cd ~
git clone https://github.com/NVlabs/curobo.git || true
cd curobo
pip install -e .

# 5. 克隆项目
echo "[5/6] 克隆项目..."
cd ~
git clone git@github.com:JJ66-git/Dobot-curobo.git || true
cd Dobot-curobo

# 6. 配置符号链接
echo "[6/6] 配置符号链接..."
ln -sf ~/Dobot-curobo/curobo/my_x_trainer ~/curobo/src/curobo/content/configs/robot/xtrainer

# 安装基础依赖
pip install numpy scipy pyyaml

echo ""
echo "=== 部署完成！==="
echo "运行测试: cd ~/Dobot-curobo/x-trainer/source/leisaac && python -m leisaac.motion_planning.test_motion_planning"
```

---

## 附录 B：测试检查清单

在调试前，确认以下各项：

- [ ] `nvidia-smi` 能看到 GPU
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` 输出 `True`
- [ ] `python -c "from curobo.wrap.reacher.ik_solver import IKSolver"` 无报错
- [ ] `ls ~/curobo/src/curobo/content/configs/robot/xtrainer/xtrainer.yml` 文件存在
- [ ] `python -m leisaac.motion_planning.test_motion_planning` 全部通过

---

## 附录 C：文件依赖关系

```
planner_interface.py  ← 统一接口（你主要调用这个）
    ├── ik_solver.py         ← cuRobo IK 求解
    ├── motion_planner.py    ← cuRobo 轨迹规划
    ├── coordinate_transform.py  ← 坐标转换
    └── task_state_machine.py    ← 状态机

planner_interface.py
    └── 需要: curobo, torch, numpy

coordinate_transform.py
    └── 需要: numpy, scipy（无 GPU 依赖）

task_state_machine.py
    └── 需要: 无外部依赖（纯 Python）
```

---

**下一步：** 在工作站上按照本文档的顺序（1→9）逐步操作。如果遇到问题，直接跳到第 8 节排查。
