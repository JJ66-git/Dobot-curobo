# Dobot X-Trainer 运动规划模块 — 工作站部署与调试指南

> 本文档指导你将 GitHub 上的新文件安全合并到工作站现有的 X-Trainer 项目中，
> 然后完成环境配置、测试、调试的全流程。
> **核心原则：不破坏现有项目结构，只新增/更新需要的文件。**

---

## 目录

1. [工作站现有项目结构](#1-工作站现有项目结构)
2. [备份现有项目](#2-备份现有项目)
3. [拉取 GitHub 上的新文件](#3-拉取-github-上的新文件)
4. [检查文件是否到位](#4-检查文件是否到位)
5. [安装 Python 依赖](#5-安装-python-依赖)
6. [配置 cuRobo 识别新机器人](#6-配置-curobo-识别新机器人)
7. [运行运动规划模块测试](#7-运行运动规划模块测试)
8. [与 Isaac Sim / policy_server 集成](#8-与-isaac-sim--policy_server-集成)
9. [常见问题排查](#9-常见问题排查)
10. [调试技巧](#10-调试技巧)

---

## 1. 工作站现有项目结构

你的工作站上已经有一个完整的 X-Trainer 项目，结构大致如下：

```
项目根目录/
├── curobo/
│   ├── my_x_trainer/          ← [新增] 机器人 URDF + cuRobo 配置
│   │   ├── build_model.py
│   │   ├── xtrainer.urdf
│   │   └── xtrainer.yml
│   └── third_party/           ← [已有] cuRobo 第三方库（不要动）
│       ├── curobo_python/
│       ├── curobo_torch/
│       └── geometric_algo/
├── docs/
│   └── LICENSE
├── policy_server/
│   └── __init__.py            ← [已有] 策略服务器
├── x-trainer/
│   └── source/leisaac/leisaac/
│       └── motion_planning/   ← [新增] 运动规划模块（7个文件）
│           ├── __init__.py
│           ├── coordinate_transform.py
│           ├── ik_solver.py
│           ├── motion_planner.py
│           ├── planner_interface.py
│           ├── task_state_machine.py
│           └── test_motion_planning.py
├── README.md
└── setup.py
```

**关键区分：**
- `[已有]` 的文件 — **不要修改**，除非你明确知道要改什么
- `[新增]` 的文件 — 从 GitHub 拉取，不会覆盖已有文件

---

## 2. 备份现有项目

**在做任何操作之前，先备份！** 这样即使出错也能恢复。

```bash
# 进入项目根目录（根据你的实际路径调整）
cd /path/to/your/x-trainer/project

# 创建备份（带日期）
cp -r . ../x-trainer-backup-$(date +%Y%m%d)

# 或者用 git 备份当前状态
git stash  # 如果有未提交的修改
git branch backup-before-merge  # 创建备份分支
```

---

## 3. 拉取 GitHub 上的新文件

### 3.1 添加 GitHub 远程仓库

```bash
cd /path/to/your/x-trainer/project

# 检查现有远程仓库
git remote -v
# 应该看到 origin 指向 gitee（或你之前用的仓库）

# 添加我们的 GitHub 仓库作为新的远程源
git remote add github https://github.com/JJ66-git/Dobot-curobo.git

# 验证
git remote -v
# 应该看到：
# origin    https://gitee.com/... (fetch)
# origin    https://gitee.com/... (push)
# github    https://github.com/JJ66-git/Dobot-curobo.git (fetch)
# github    https://github.com/JJ66-git/Dobot-curobo.git (push)
```

### 3.2 先看有哪些新文件（不要直接合并！）

```bash
# 查看 GitHub 分支和本地分支的差异
git fetch github

# 查看 GitHub 上有哪些文件是我们本地没有的
git diff master github/master --stat
```

输出会告诉你哪些文件是新增的、哪些是修改的。

### 3.3 只拉取需要的文件（安全方式）

**不要直接 `git merge`！** 那样可能会覆盖你现有的文件。我们只提取需要的新文件：

```bash
# === 新增文件 1：curobo/my_x_trainer/ ===
# 这个目录在你的项目中可能还不存在，直接检出

git checkout github/master -- curobo/my_x_trainer/build_model.py
git checkout github/master -- curobo/my_x_trainer/xtrainer.urdf
git checkout github/master -- curobo/my_x_trainer/xtrainer.yml

# === 新增文件 2：motion_planning 模块 ===
git checkout github/master -- x-trainer/source/leisaac/leisaac/motion_planning/__init__.py
git checkout github/master -- x-trainer/source/leisaac/leisaac/motion_planning/coordinate_transform.py
git checkout github/master -- x-trainer/source/leisaac/leisaac/motion_planning/ik_solver.py
git checkout github/master -- x-trainer/source/leisaac/leisaac/motion_planning/motion_planner.py
git checkout github/master -- x-trainer/source/leisaac/leisaac/motion_planning/planner_interface.py
git checkout github/master -- x-trainer/source/leisaac/leisaac/motion_planning/task_state_machine.py
git checkout github/master -- x-trainer/source/leisaac/leisaac/motion_planning/test_motion_planning.py

# === 可选文件 ===
git checkout github/master -- .gitignore
git checkout github/master -- MOTION_PLANNING_TUTORIAL.md
git checkout github/master -- MOTION_PLANNING_DEPLOY_GUIDE.md
```

> **这些命令的含义：** 从 `github/master` 分支中取出指定文件，放到你的工作目录中。
> **不会删除** 你现有的任何文件，只会新增或更新列出的文件。

### 3.4 提交这些新文件

```bash
# 查看状态
git status
# 你应该只看到新增/修改的文件，不会看到任何删除

# 提交
git commit -m "Add motion planning module for X-Trainer dual-arm robot"
```

---

## 4. 检查文件是否到位

运行以下命令，确认所有文件都已就位：

```bash
# 检查 curobo 配置
ls -la curobo/my_x_trainer/
# 应该有: build_model.py  xtrainer.urdf  xtrainer.yml

# 检查 motion_planning 模块
ls -la x-trainer/source/leisaac/leisaac/motion_planning/
# 应该有 7 个 .py 文件

# 检查 .gitignore（确保 curobo/third_party/ 不会被 git 跟踪）
cat .gitignore
```

如果某个目录或文件不存在，检查路径是否正确。你的项目根目录可能不在 `~/Dobot-curobo/`，而是在其他位置。

---

## 5. 安装 Python 依赖

### 5.1 检查现有环境

```bash
# 查看当前 Python 环境
which python3
python3 --version

# 检查 PyTorch 是否已安装且支持 CUDA
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

如果 PyTorch 已安装且 `CUDA: True`，跳到 5.3。

### 5.2 安装 PyTorch（如果没有）

```bash
# 创建 conda 环境（如果还没有）
conda create -n curobo python=3.10 -y
conda activate curobo

# 安装 PyTorch（根据你的 CUDA 版本选择）
# 查看 CUDA 版本: nvidia-smi
# CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 5.3 安装项目依赖

```bash
pip install numpy scipy pyyaml
```

### 5.4 验证 cuRobo 已可用

```bash
python3 -c "from curobo.wrap.reacher.ik_solver import IKSolver; print('cuRobo OK')"
```

如果报错 `ModuleNotFoundError: No module named 'curobo'`，说明 cuRobo 还没安装：

```bash
# 进入项目的 third_party 目录
cd curobo/third_party/curobo_python
pip install -e .
cd ../../..
```

或者如果 cuRobo 是独立安装的：

```bash
cd ~/curobo  # 你之前安装 cuRobo 的位置
pip install -e .
```

---

## 6. 配置 cuRobo 识别新机器人

cuRobo 需要知道 `xtrainer.yml` 在哪里。

### 方式 A：符号链接（推荐）

```bash
# 找到 cuRobo 的配置目录
# 如果 cuRobo 是 pip install 安装的：
CUROBO_CONFIG_DIR=$(python3 -c "import curobo; import os; print(os.path.join(os.path.dirname(curobo.__file__), 'content', 'configs', 'robot'))")
echo "cuRobo 配置目录: $CUROBO_CONFIG_DIR"

# 创建符号链接
ln -sf "$(pwd)/curobo/my_x_trainer" "$CUROBO_CONFIG_DIR/xtrainer"

# 验证
ls -la "$CUROBO_CONFIG_DIR/xtrainer"
# 应该指向 curobo/my_x_trainer/ 并看到 xtrainer.yml
```

### 方式 B：直接指定绝对路径

如果符号链接不方便，可以在代码中直接用绝对路径：

```python
# 在你的代码中
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF_PATH = os.path.join(PROJECT_ROOT, "curobo", "my_x_trainer", "xtrainer.urdf")
YAML_PATH = os.path.join(PROJECT_ROOT, "curobo", "my_x_trainer", "xtrainer.yml")
```

### 验证 cuRobo 能加载机器人

```bash
cd /path/to/your/x-trainer/project

python3 -c "
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
try:
    config = IKSolverConfig.from_robot_yaml('xtrainer.yml', num_seeds=4, self_collision_check=True)
    solver = IKSolver(config)
    print('[OK] cuRobo 成功加载 xtrainer.yml')
except Exception as e:
    print(f'[FAIL] 加载失败: {e}')
"
```

---

## 7. 运行运动规划模块测试

### 7.1 运行完整测试套件

```bash
cd /path/to/your/x-trainer/project/x-trainer/source/leisaac
python3 -m leisaac.motion_planning.test_motion_planning
```

### 7.2 预期输出

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

### 7.3 单独运行某个测试（调试用）

```bash
python3 -c "
from leisaac.motion_planning.test_motion_planning import test_ik_solver
test_ik_solver()
"
```

---

## 8. 与 Isaac Sim / policy_server 集成

### 8.1 在 policy_server 中引入运动规划模块

在你现有的 `policy_server/__init__.py`（或对应的 server 文件）中添加：

```python
# 在文件顶部添加导入
import sys
import os

# 确保 motion_planning 模块可以被找到
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "x-trainer", "source", "leisaac"))

from leisaac.motion_planning import MotionPlanningModule

# 在你的 PolicyServer 类中初始化
class PolicyServer:
    def __init__(self):
        # ... 你现有的初始化代码 ...

        # 添加运动规划模块
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

### 8.2 在 Isaac Sim 中使用

```python
from leisaac.motion_planning import MotionPlanningModule, SceneBuilder

module = MotionPlanningModule()
module.warmup()

# 设置障碍物
scene = SceneBuilder()
scene.add_table("work_table", position=[0.5, 0.0, 0.0], dimensions=[0.8, 1.0, 0.02])

# 规划
target = {"position": [0.3, 0.2, 0.3], "quaternion": [1, 0, 0, 0]}
result = module.plan(target, obstacles=scene.get_obstacles(), arm="left")

if result.success:
    # result.trajectory 是关节角度轨迹，可以直接发送给仿真机器人执行
    for waypoint in result.trajectory:
        # 设置关节位置
        pass
```

---

## 9. 常见问题排查

### 9.1 `fatal: pathspec did not match any files`

**原因：** GitHub 上的文件路径和你本地的项目路径不一致。

```bash
# 先看看 GitHub 上有什么
git ls-tree -r --name-only github/master

# 如果路径不同，手动创建目录再检出
mkdir -p x-trainer/source/leisaac/leisaac/motion_planning
git checkout github/master -- x-trainer/source/leisaac/leisaac/motion_planning/
```

### 9.2 `CUDA not available` / `torch.cuda.is_available()` 返回 False

```bash
nvidia-smi                           # 检查 GPU 驱动
python3 -c "import torch; print(torch.version.cuda)"  # 检查 PyTorch CUDA 版本

# 如果不匹配，重装 PyTorch
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 9.3 `ModuleNotFoundError: No module named 'curobo'`

```bash
# 方式 1：通过项目内的 third_party 安装
cd curobo/third_party/curobo_python
pip install -e .

# 方式 2：如果你之前独立安装过 cuRobo
cd ~/curobo
pip install -e .
```

### 9.4 `ModuleNotFoundError: No module named 'leisaac'`

```bash
# 确保从正确的目录运行
cd /path/to/your/x-trainer/project/x-trainer/source/leisaac
python3 -m leisaac.motion_planning.test_motion_planning

# 或者设置 PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/path/to/your/x-trainer/project/x-trainer/source/leisaac
```

### 9.5 IK 求解全部失败（success=False）

```bash
# 快速诊断：测试一个在工作空间内的近处目标
python3 -c "
from leisaac.motion_planning import MotionPlanningModule
m = MotionPlanningModule()
m.warmup()
r = m.plan({'position': [0.2, 0.1, 0.2], 'quaternion': [1,0,0,0]}, [], 'left')
print(r)
"
```

如果仍然失败，检查：
1. `xtrainer.urdf` 中的关节限位是否合理
2. 运行 `python3 curobo/my_x_trainer/build_model.py` 重新生成配置
3. 启用日志：`logging.getLogger("curobo").setLevel(logging.DEBUG)`

### 9.6 `FileNotFoundError: xtrainer.yml not found`

```bash
# 确认文件存在
ls curobo/my_x_trainer/xtrainer.yml

# 确认符号链接正确
CUROBO_CONFIG_DIR=$(python3 -c "import curobo,os;print(os.path.join(os.path.dirname(curobo.__file__),'content','configs','robot'))")
ls -la "$CUROBO_CONFIG_DIR/xtrainer"
```

### 9.7 规划失败（plan_to_pose 返回 success=False）

可能原因：
1. **碰撞** — 目标位姿或路径上有障碍物
2. **关节限位** — 路径上某些关节超出限位
3. **规划超时** — 增加规划时间

```python
# 增加规划时间
from leisaac.motion_planning.motion_planner import DualArmMotionPlanner
planner = DualArmMotionPlanner(max_time=5.0)  # 默认 2 秒
```

---

## 10. 调试技巧

### 10.1 打印 IK 求解详细日志

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("curobo").setLevel(logging.DEBUG)
```

### 10.2 导出轨迹为 CSV

```python
import csv
result = module.plan(target_pose, obstacles=[], arm="left")
if result.success:
    with open("trajectory.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "j1", "j2", "j3", "j4", "j5", "j6"])
        for i, q in enumerate(result.trajectory):
            writer.writerow([i * 0.02] + q)
    print("轨迹已保存到 trajectory.csv")
```

### 10.3 用 matplotlib 可视化关节轨迹

```python
import matplotlib.pyplot as plt
import csv

times, joints = [], []
with open("trajectory.csv") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        times.append(float(row[0]))
        joints.append([float(x) for x in row[1:]])

joints = list(zip(*joints))
fig, axes = plt.subplots(3, 2, figsize=(12, 8))
for i, ax in enumerate(axes.flat):
    ax.plot(times, joints[i])
    ax.set_title(f"Joint {i+1}")
    ax.set_ylabel("Angle (rad)")
    ax.grid(True)
plt.tight_layout()
plt.savefig("joint_trajectory.png", dpi=150)
```

### 10.4 性能分析

```python
import time
module.warmup()
target = {"position": [0.3, 0.2, 0.3], "quaternion": [1, 0, 0, 0]}

times = []
for _ in range(100):
    start = time.perf_counter()
    module.plan(target, [], "left")
    times.append(time.perf_counter() - start)

print(f"平均: {sum(times)/len(times)*1000:.1f} ms")
print(f"最快: {min(times)*1000:.1f} ms")
print(f"最慢: {max(times)*1000:.1f} ms")
```

---

## 附录 A：快速操作清单（TL;DR）

```bash
# 1. 备份
cd /path/to/project
cp -r . ../backup-$(date +%Y%m%d)

# 2. 添加远程仓库
git remote add github https://github.com/JJ66-git/Dobot-curobo.git
git fetch github

# 3. 拉取新文件
git checkout github/master -- curobo/my_x_trainer/
git checkout github/master -- x-trainer/source/leisaac/leisaac/motion_planning/
git checkout github/master -- .gitignore

# 4. 提交
git commit -m "Add motion planning module for X-Trainer"

# 5. 配置 cuRobo
ln -sf "$(pwd)/curobo/my_x_trainer" "$(python3 -c 'import curobo,os;print(os.path.join(os.path.dirname(curobo.__file__),"content","configs","robot"))')/xtrainer"

# 6. 运行测试
cd x-trainer/source/leisaac
python3 -m leisaac.motion_planning.test_motion_planning
```

---

## 附录 B：文件依赖关系

```
planner_interface.py  ← 统一接口（主要调用这个）
    ├── ik_solver.py         ← cuRobo IK 求解（需要 GPU）
    ├── motion_planner.py    ← cuRobo 轨迹规划（需要 GPU）
    ├── coordinate_transform.py  ← 坐标转换（无 GPU 依赖）
    └── task_state_machine.py    ← 状态机（无 GPU 依赖）
```

## 附录 C：测试检查清单

- [ ] `nvidia-smi` 能看到 GPU
- [ ] `python3 -c "import torch; print(torch.cuda.is_available())"` 输出 `True`
- [ ] `python3 -c "from curobo.wrap.reacher.ik_solver import IKSolver"` 无报错
- [ ] `ls curobo/my_x_trainer/xtrainer.yml` 文件存在
- [ ] `python3 -m leisaac.motion_planning.test_motion_planning` 全部通过

---

**下一步：** 按照第 2 节（备份）→ 第 3 节（拉取）→ 第 4 节（检查）的顺序操作。
