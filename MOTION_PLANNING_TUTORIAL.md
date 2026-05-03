# Dobot 双臂抓取赛 — 运动规划模块开发教程

> 适用对象：大一学生，只会 Python/C++ 基础语法，无 cuRobo / 运动规划经验
> 最终目标：形成可对接 `policy_server` 的运动规划模块，输入末端位姿 → 输出双臂关节轨迹

---

## 总体架构：感知 → 决策 → 坐标转换 → IK 规划 → 动作分发

```
┌─────────────┐    ┌──────────────┐    ┌──────────────────┐    ┌─────────────┐    ┌──────────────┐
│  相机感知    │ →  │ 坐标转换     │ →  │ 任务状态机       │ →  │ cuRobo IK   │ →  │ 动作分发     │
│ (RGB图像)   │    │ 相机→基座    │    │ + 路径中间点     │    │ + 避障规划   │    │ (关节轨迹)  │
└─────────────┘    └──────────────┘    └──────────────────┘    └─────────────┘    └──────────────┘
```

**你负责的模块**（本教程覆盖）：
1. **坐标转换**：把相机看到的目标位置从相机坐标系转换到机器人基座坐标系
2. **任务状态机**：管理抓取任务的阶段（接近 → 抓取 → 提升 → 放置）
3. **cuRobo 集成**：逆运动学求解 + 避障 + 轨迹生成

---

## 目录

| 步骤 | 内容 | 预计时间 |
|------|------|---------|
| Step 0 | 环境验证 & 项目结构认知 | 30 分钟 |
| Step 1 | 创建 X-Trainer URDF（给 cuRobo 用） | 1 小时 |
| Step 2 | 用 cuRobo 构建机器人模型（碰撞球 + 配置文件） | 30 分钟 |
| Step 3 | 坐标转换模块：相机 → 基座 | 1 小时 |
| Step 4 | 任务状态机 + 路径中间点管理 | 1.5 小时 |
| Step 5 | cuRobo IK 逆运动学求解 | 1 小时 |
| Step 6 | cuRobo 避障 + 轨迹生成 | 1.5 小时 |
| Step 7 | 集成到 policy_server 对接接口 | 1 小时 |
| Step 8 | 端到端测试 & 调参 | 2 小时 |

---

## Step 0：环境验证 & 项目结构认知

### 你要干什么

确认你的电脑上所有依赖都装好了，搞清楚文件都在哪里。

### 为什么这么做

很多同学卡在"跑不起来"这一步。先验证环境，后面每一步都能跑通。

### 需要注意什么

- cuRobo 需要 **NVIDIA GPU + CUDA**，确认你有 N 卡
- Python 版本建议 3.10+
- 所有命令在项目根目录 `C:\jj\Dobot\Guangdong_Collegiate_Computing_Competition` 下执行

### 执行命令

```bash
# 1. 确认 Python 和 CUDA
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 2. 确认 cuRobo 已安装
python -c "import curobo; print(f'cuRobo version: {curobo.__version__}')"

# 3. 确认项目结构
ls curobo/content/configs/robot/        # 应该看到 franka.yml, dual_ur10e.yml 等
ls x-trainer/source/leisaac/leisaac/    # 应该看到 tasks/, policy/, utils/ 等
```

### 发给你的子 Prompt（复制这段发给我即可）

> **【子 Prompt - Step 0 环境检查】**
> 请帮我检查当前环境是否满足运动规划模块开发需求：
> 1. 检查 Python 版本 >= 3.10
> 2. 检查 PyTorch 是否安装且 CUDA 可用
> 3. 检查 cuRobo 是否可 import
> 4. 检查 curobo/content/configs/robot/ 目录下有哪些机器人配置文件
> 5. 检查 x-trainer/source/leisaac/leisaac/utils/constant.py 中的关节名称定义
> 如果有缺失，告诉我怎么装。

---

## Step 1：创建 X-Trainer 的 URDF 文件（给 cuRobo 用）

### 你要干什么

把 Dobot X-Trainer 双臂机器人的运动学模型写成 cuRobo 能读取的 URDF 格式。

### 为什么这么做

cuRobo 需要 URDF 文件来理解机器人的关节结构、连杆长度、关节限位等信息。
你的项目里有 USD 格式的模型（给 Isaac Sim 用），但 cuRobo 不认 USD，只认 URDF。

### 关键概念解释（给大一同学）

- **URDF**（Unified Robot Description Format）：一种用 XML 描述机器人的标准格式
  - `<link>` = 刚体部件（如手臂、底座）
  - `<joint>` = 关节（连接两个 link，可以转动）
  - 每个关节有：类型（旋转/固定）、父 link、子 link、转轴方向、限位角度
- **为什么需要这个文件**：cuRobo 的 IK 求解器需要知道"关节怎么连的"才能算出"关节转多少度能到达目标位置"

### X-Trainer 双臂关节结构

根据 `xtrainer.py` 中的定义，X-Trainer 有 16 个关节：

| 关节名 | 所属手臂 | 功能 |
|--------|---------|------|
| J1_1 ~ J1_6 | 左臂 | 6 自由度（肩2 + 肘1 + 腕3） |
| J1_7, J1_8 | 左手 | 夹爪开合 |
| J2_1 ~ J2_6 | 右臂 | 6 自由度 |
| J2_7, J2_8 | 右手 | 夹爪开合 |

**cuRobo 只需要规划 6DOF 手臂**（J1_1~J1_6 和 J2_1~J2_6），夹爪（J1_7/J1_8/J2_7/J2_8）单独控制。

### 需要注意什么

- URDF 中的连杆长度需要从实际模型中测量（单位：米）
- 如果你不确定具体尺寸，先用下面的模板（基于合理的估计），后续可以校准
- 夹爪关节在 cuRobo 配置中会被"锁定"（lock_joints），不参与 IK 求解

### 发给你的子 Prompt

> **【子 Prompt - Step 1 创建 URDF】**
> 我需要为 Dobot X-Trainer 双臂机器人创建 cuRobo 兼容的 URDF 文件。
>
> 已知信息：
> - 双臂各 6 个自由度（不含夹爪），关节名 J1_1~J1_6（左臂）、J2_1~J2_6（右臂）
> - 夹爪关节 J1_7/J1_8（左）、J2_7/J2_8（右），cuRobo 规划时不参与
> - 关节限位均为 -3.14159 ~ 3.14159 弧度（±180°）
> - 机器人基座固定在世界坐标系原点
> - 左臂基座相对机器人中心偏移约 (-0.075, 0, 0)，右臂偏移约 (0.075, 0, 0)
>
> 请在 `curobo/my_x_trainer/` 目录下创建 `xtrainer.urdf` 文件，要求：
> 1. 包含 world_base_link（世界基座）+ 左臂 6 连杆 + 右臂 6 连杆 + 左右末端执行器
> 2. 每个连杆有合理的质量（0.1~2.0 kg）和惯性矩阵（可以用圆柱体近似估算）
> 3. 关节类型全部为 revolute（旋转关节）
> 4. 包含夹爪关节但标记为 fixed（cuRobo 中会锁定）
> 5. 注释详细，每行关键代码都有中文说明
> 6. 参考 curobo/content/assets/robot/ur_description/dual_ur10e.urdf 的结构

---

## Step 2：用 cuRobo 构建机器人模型（碰撞球 + 配置 YAML）

### 你要干什么

用 cuRobo 的 `RobotBuilder` 工具，从 Step 1 的 URDF 自动生成：
1. **碰撞球**（每个连杆用一组球体近似，用于避障计算）
2. **自碰撞忽略矩阵**（哪些连杆对不需要检查碰撞）
3. **YAML 配置文件**（cuRobo 所有模块都要读这个文件）

### 为什么这么做

cuRobo 在 GPU 上做碰撞检测时，不用真实网格（太慢），而是用球体近似每个连杆。
这一步就是自动帮你算"每个连杆要几个球、球放哪里、半径多大"。

### 关键概念

- **碰撞球（Collision Spheres）**：每个连杆用 N 个球体包裹，球越多越精确但越慢
- **自碰撞忽略矩阵**：相邻连杆（如 shoulder 和 upper_arm）永远在碰撞，需要忽略
- **YAML 配置文件**：cuRobo 的"身份证"，后续 IK、规划、避障都要读它

### 需要注意什么

- 这一步需要 GPU，确认 CUDA 可用
- 生成的 YAML 文件会保存到你指定的路径，后续所有步骤都引用它
- 如果球体拟合质量不好（coverage < 90%），可以调高 `sphere_density`

### 发给你的子 Prompt

> **【子 Prompt - Step 2 构建机器人模型】**
> 请帮我用 cuRobo 的 RobotBuilder 为 Step 1 创建的 URDF 生成机器人配置文件。
>
> 要求：
> 1. 运行 `curobo.robot_builder.RobotBuilder` 从 URDF 生成碰撞球和自碰撞矩阵
> 2. 工具帧（tool_frames）设为左右末端执行器：["left_ee_link", "right_ee_link"]
> 3. sphere_density 设为 1.5（稍微密一点，提高精度）
> 4. 输出 YAML 文件保存到 `curobo/my_x_trainer/xtrainer.yml`
> 5. 同时输出拟合质量指标（coverage、protrusion 等）
> 6. 代码中每个关键步骤都加中文注释
> 7. 提供一个独立的 Python 脚本 `curobo/my_x_trainer/build_model.py`，可以直接 `python build_model.py` 运行
>
> 参考 cuRobo 官方示例：`curobo/examples/getting_started/build_robot_model.py`

---

## Step 3：坐标转换模块（相机 → 基座参考系）

### 你要干什么

把相机检测到的目标物体位置（相机坐标系下的 xyz）转换到机器人基座坐标系下。

### 为什么这么做

- 相机看到的目标位置是在**相机坐标系**下（以相机光心为原点）
- 机器人 IK 求解需要的是**基座坐标系**下的位置（以机器人底座为原点）
- 需要经过矩阵变换：`P_base = T_base_camera × P_camera`

### 关键概念（给大一同学）

- **坐标系（Frame）**：每个物体都有自己的"参考系"，就像每个人有自己的"前后左右"
- **变换矩阵（Transform Matrix）**：4×4 矩阵，描述一个坐标系相对于另一个坐标系的位置和姿态
  ```
  T = [R  t]    R = 3×3 旋转矩阵（描述朝向）
      [0  1]    t = 3×1 平移向量（描述位置）
  ```
- **链式变换**：相机 → 末端 → 基座，需要把多个变换矩阵连乘

### X-Trainer 的相机配置

根据 `xtrainer_arm_env_cfg.py`，有 3 个相机：

| 相机名 | 安装位置 | 用途 |
|--------|---------|------|
| left_wrist | 左臂末端 J1_6 | 手眼相机，近距离操作 |
| right_wrist | 右臂末端 J2_6 | 手眼相机，近距离操作 |
| top | 机器人基座 base_link | 俯视相机，全局视野 |

### 需要注意什么

- 相机安装偏移量在 `xtrainer_arm_env_cfg.py` 中有定义（pos 和 rot）
- 旋转用的是 ROS 约定（wxyz 四元数）
- 如果用 RealSense 真实相机，还需要考虑手眼标定误差

### 发给你的子 Prompt

> **【子 Prompt - Step 3 坐标转换】**
> 请帮我创建相机坐标到机器人基座坐标的转换模块。
>
> 已知信息：
> - 机器人基座坐标系：base_link，原点在机器人中心
> - 左腕相机安装偏移：pos=(0.0, -0.065, 0.03)，rot=euler(-15°, 0, 0)，安装在 J1_6 连杆上
> - 右腕相机安装偏移：pos=(0.0, -0.065, 0.03)，rot=euler(-15°, 0, 0)，安装在 J2_6 连杆上
> - 俯视相机安装偏移：pos=(0.53, -0.55, 1.0)，rot=euler(-148°, 0, 0)，安装在 base_link 上
> - 相机内参：focal_length=26.8, horizontal_aperture=36.83, 分辨率 640×480
>
> 请在 `x-trainer/source/leisaac/leisaac/motion_planning/` 目录下创建 `coordinate_transform.py`，要求：
> 1. 实现 `CameraToBaseTransformer` 类
> 2. 支持从像素坐标 (u, v) + 深度值 → 3D 相机坐标 → 基座坐标
> 3. 支持 wrist_camera（需要当前关节角度来算末端位姿）和 top_camera（固定变换）
> 4. 使用 numpy 实现，不依赖复杂库
> 5. 每个函数都有详细的中文注释，解释矩阵运算的含义
> 6. 包含一个简单的测试函数，验证转换是否正确
>
> 关键公式：
> - 像素→相机坐标：x = (u - cx) * z / fx, y = (v - cy) * z / fy
> - 相机→基座：P_base = T_base_ee @ T_ee_camera @ P_camera

---

## Step 4：任务状态机 + 路径中间点管理

### 你要干什么

实现一个"状态机"来管理抓取任务的流程：检测目标 → 移动到目标上方 → 下降 → 抓取 → 提升 → 移动到放置区 → 下降 → 放置 → 回到初始位置。

### 为什么这么做

- 双臂抓取不是"一步到位"的，需要分多个阶段
- 每个阶段有不同的目标位姿和约束（如抓取阶段要慢、提升阶段要垂直）
- 状态机让你的代码清晰、可调试、不容易出错

### 关键概念

- **状态机（State Machine）**：程序在任意时刻处于一个"状态"，满足条件后跳到下一个状态
  ```
  IDLE → MOVE_ABOVE → DESCEND → GRASP → LIFT → MOVE_TO_PLACE → PLACE_DESCEND → RELEASE → RETURN
  ```
- **中间点（Waypoint）**：从 A 到 B 不是直线，而是经过一系列"路标点"
  - 好处：避障、运动可预测、安全性高
  - 例如：抓取前先移到目标正上方 10cm，再垂直下降

### 需要注意什么

- 状态机要支持"失败回退"（如 IK 求解失败时回到上一个安全状态）
- 左右臂可能需要并行执行不同状态（如左臂抓取时右臂在等待）
- 中间点的高度偏移量需要根据实际任务调整

### 发给你的子 Prompt

> **【子 Prompt - Step 4 任务状态机】**
> 请帮我创建双臂抓取任务的状态机和中间点管理模块。
>
> 要求：
> 1. 在 `x-trainer/source/leisaac/leisaac/motion_planning/` 目录下创建 `task_state_machine.py`
> 2. 实现 `GraspTaskStateMachine` 类，包含以下状态：
>    - IDLE（空闲）
>    - DETECT（检测目标位置）
>    - MOVE_ABOVE（移动到目标上方，z 方向偏移 +10cm）
>    - DESCEND（垂直下降到抓取位置）
>    - GRASP（闭合夹爪）
>    - LIFT（垂直提升，z 方向偏移 +15cm）
>    - TRANSIT（移动到放置区上方）
>    - PLACE_DESCEND（下降到放置位置）
>    - RELEASE（打开夹爪）
>    - RETREAT（后撤回到安全位置）
>    - DONE（完成）
>    - ERROR（出错，需要回退）
> 3. 每个状态有：目标位姿、进入条件、超时处理、失败回退逻辑
> 4. 实现 `WaypointManager` 类，管理中间点：
>    - `add_waypoint(pose, description)` — 添加中间点
>    - `get_next_waypoint()` — 获取下一个中间点
>    - `generate_grasp_waypoints(target_pose, approach_offset=0.10)` — 自动生成抓取路径的中间点序列
> 5. 支持双臂独立状态（左臂和右臂可以处于不同状态）
> 6. 所有代码用中文注释
>
> 数据结构示例：
> ```python
> @dataclass
> class Waypoint:
>     position: np.ndarray      # [x, y, z] 位置
>     quaternion: np.ndarray    # [qw, qx, qy, qz] 姿态四元数
>     gripper_state: float      # 0.0=闭合, 1.0=张开
>     speed_factor: float       # 速度系数 0~1
>     description: str          # 中文描述
> ```

---

## Step 5：cuRobo IK 逆运动学求解

### 你要干什么

用 cuRobo 的 `InverseKinematics` 模块，输入目标末端位姿（xyz + 四元数），输出关节角度。

### 为什么这么做

- 你的策略模型输出的是"末端执行器应该去哪里"（xyz 坐标）
- 但机器人需要的是"每个关节转多少度"（关节角度）
- IK 就是做这个转换的：`末端位姿 → 关节角度`

### 关键概念

- **正运动学（FK）**：已知关节角度 → 算末端位置（cuRobo 的 `Kinematics` 模块）
- **逆运动学（IK）**：已知末端位置 → 算关节角度（cuRobo 的 `InverseKinematics` 模块）
- **多解问题**：同一个末端位置可能对应多组关节角度，cuRobo 用 GPU 并行求解多个种子，选最优解
- **cuRobo 的 IK 特点**：
  - GPU 加速：同时求解 32~256 个种子，选最优
  - 碰撞感知：自动避开自碰撞
  - 高精度：位置误差 < 1mm

### API 调用流程

```python
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.types import JointState, Pose, GoalToolPose

# 1. 创建 IK 求解器（读取 Step 2 生成的 YAML 配置）
config = InverseKinematicsCfg.create(robot="my_x_trainer/xtrainer.yml", num_seeds=64)
ik = InverseKinematics(config)

# 2. 定义目标位姿
target = Pose(
    position=torch.tensor([[x, y, z]]),        # 目标位置
    quaternion=torch.tensor([[qw, qx, qy, qz]]) # 目标姿态
)

# 3. 求解 IK
result = ik.solve_pose(GoalToolPose.from_poses({tool_frame: target}, num_goalset=1))

# 4. 检查结果
if result.success.item():
    joint_angles = result.js_solution.position  # 关节角度（弧度）
```

### 需要注意什么

- `num_seeds` 越多成功率越高，但越慢（32~128 是合理范围）
- 四元数顺序是 **wxyz**（cuRobo 约定），不是 xyzw
- 如果 IK 求解失败，可以尝试：
  - 增加 num_seeds
  - 调整目标位姿（可能超出工作空间）
  - 检查关节限位

### 发给你的子 Prompt

> **【子 Prompt - Step 5 IK 求解】**
> 请帮我创建 cuRobo IK 逆运动学求解模块。
>
> 要求：
> 1. 在 `x-trainer/source/leisaac/leisaac/motion_planning/` 目录下创建 `ik_solver.py`
> 2. 实现 `DualArmIKSolver` 类，支持：
>    - 初始化时加载 Step 2 生成的 `curobo/my_x_trainer/xtrainer.yml` 配置
>    - `solve_left_arm(target_pose)` — 左臂 IK 求解
>    - `solve_right_arm(target_pose)` — 右臂 IK 求解
>    - `solve_both(left_pose, right_pose)` — 双臂同时求解
>    - `check_ik_valid(joint_angles, arm)` — 验证关节角度是否在限位内
> 3. target_pose 格式：`{"position": [x,y,z], "quaternion": [qw,qx,qy,qz]}`
> 4. 返回值包含：成功标志、关节角度（list[float]）、位置误差（mm）
> 5. IK 求解失败时提供友好错误信息（如"目标超出左臂工作空间"）
> 6. 包含 `forward_kinematics(joint_angles, arm)` 方法用于验证
> 7. 所有代码用中文注释
>
> 关键参考：
> - cuRobo 官方示例：`curobo/examples/getting_started/inverse_kinematics.py`
> - API 入口：`from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg`

---

## Step 6：cuRobo 避障 + 轨迹生成

### 你要干什么

用 cuRobo 的 `MotionPlanner` 模块，从当前关节状态规划到目标关节状态的**无碰撞轨迹**。

### 为什么这么做

- IK 只告诉你"终点"在哪里，不告诉你"怎么过去"
- 轨迹规划负责生成从 A 到 B 的平滑路径，同时避开障碍物
- 比赛中有桌子、物体等障碍物，必须用避障规划

### 关键概念

- **轨迹优化（Trajectory Optimization）**：cuRobo 把路径表示为一系列"节点点"，同时优化平滑性和避障
- **碰撞场景（Scene）**：用立方体、球体等简单几何体描述障碍物
- **运动规划器（MotionPlanner）**：cuRobo 的高级 API，一键完成"从 A 到 B 的无碰撞规划"
- **插值（Interpolation）**：规划器输出的节点点比较稀疏，插值后得到密集的关节轨迹

### API 调用流程

```python
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import JointState, GoalToolPose, Pose

# 1. 创建规划器
config = MotionPlannerCfg.create(
    robot="my_x_trainer/xtrainer.yml",
    scene_model="collision_table.yml",  # 障碍物场景
)
planner = MotionPlanner(config)
planner.warmup(enable_graph=True)  # 预热 CUDA 图

# 2. 定义起始状态和目标
q_start = JointState.from_position(current_joints, joint_names=planner.joint_names)
goal_pose = GoalToolPose(...)

# 3. 规划
result = planner.plan_pose(goal_pose, q_start)

# 4. 获取轨迹
if result.success.any():
    trajectory = result.get_interpolated_plan()  # 插值后的密集轨迹
```

### 需要注意什么

- `warmup()` 第一次调用会编译 CUDA 图，需要几秒；后续调用很快
- 障碍物场景需要你根据比赛场地手动定义（桌子、篮子等）
- 轨迹的 `dt`（时间步长）决定了运动速度，一般 0.02~0.05 秒

### 发给你的子 Prompt

> **【子 Prompt - Step 6 避障规划】**
> 请帮我创建 cuRobo 运动规划模块，实现避障轨迹生成。
>
> 要求：
> 1. 在 `x-trainer/source/leisaac/leisaac/motion_planning/` 目录下创建 `motion_planner.py`
> 2. 实现 `DualArmMotionPlanner` 类，支持：
>    - 初始化时加载机器人配置和场景配置
>    - `plan_to_pose(arm, target_pose, current_joints, obstacles=None)` — 单臂规划到目标位姿
>    - `plan_joint_to_joint(arm, start_joints, goal_joints)` — 关节空间到关节空间规划
>    - `plan_grasp(arm, grasp_pose, current_joints)` — 三阶段抓取规划（接近→抓取→提升）
>    - `update_obstacles(obstacles)` — 运行时更新障碍物
> 3. 实现 `SceneBuilder` 辅助类，用于快速构建比赛场景：
>    - `add_table(name, position, dimensions)` — 添加桌子
>    - `add_box(name, position, dimensions)` — 添加盒子
>    - `add_basket(name, position, dimensions)` — 添加篮子
> 4. 返回值：成功标志、关节轨迹（list of list[float]）、轨迹时长（秒）
> 5. 包含轨迹插值功能（dt=0.02 秒），输出密集轨迹点
> 6. 所有代码用中文注释
>
> 关键参考：
> - cuRobo 官方示例：`curobo/examples/getting_started/motion_planning.py`
> - 场景配置：`curobo/content/configs/scene/collision_test.yml`
> - API 入口：`from curobo.motion_planner import MotionPlanner, MotionPlannerCfg`

---

## Step 7：集成到 policy_server 对接接口

### 你要干什么

把前面所有模块组合成一个统一的接口，能被 `policy_server` 调用。

### 为什么这么做

比赛的流水线是：`策略模型 → policy_server → 你的运动规划模块 → 关节轨迹 → 仿真执行`
你需要提供一个"黑盒"接口：输入目标位姿，输出关节轨迹。

### 接口设计

```python
class MotionPlanningModule:
    """运动规划模块 — 对接 policy_server 的统一接口"""

    def plan(self, target_pose: dict, obstacles: list = None) -> dict:
        """
        输入：
            target_pose: {"position": [x,y,z], "quaternion": [qw,qx,qy,qz]}
            obstacles: [{"name": "table", "position": [x,y,z], "size": [w,h,d]}, ...]
        输出：
            {
                "success": True/False,
                "trajectory": [[j1,j2,...,j6], ...],  # 关节角度序列
                "duration": 2.5,                        # 轨迹时长（秒）
                "gripper_states": [0.0, 0.0, ...],     # 每步夹爪状态
                "error_msg": ""                          # 失败原因
            }
        """
```

### 需要注意什么

- 接口要简洁，不要暴露 cuRobo 内部细节
- 要处理各种异常（IK 失败、规划失败、超时等）
- 返回的轨迹格式要和 `policy_server` 期望的一致

### 发给你的子 Prompt

> **【子 Prompt - Step 7 集成接口】**
> 请帮我创建运动规划模块的统一接口，对接 policy_server。
>
> 要求：
> 1. 在 `x-trainer/source/leisaac/leisaac/motion_planning/` 目录下创建 `planner_interface.py`
> 2. 实现 `MotionPlanningModule` 类，整合前面所有模块：
>    - 初始化：加载 IK 求解器、运动规划器、坐标转换器
>    - `plan(target_pose, obstacles, arm="left")` — 核心接口
>    - `plan_grasp_sequence(target_pose, place_pose, obstacles)` — 完整抓取序列规划
>    - `get_current_state()` — 获取当前关节状态
> 3. plan() 的流程：
>    a. 检查目标是否在工作空间内
>    b. 调用 IK 求解目标关节角度
>    c. 调用运动规划器生成无碰撞轨迹
>    d. 插值并返回密集轨迹
> 4. 异常处理要完善：IK 失败、规划失败、超时都要返回友好错误信息
> 5. 支持 JSON 序列化（方便 gRPC 传输）
> 6. 所有代码用中文注释
>
> 同时创建 `x-trainer/source/leisaac/leisaac/motion_planning/__init__.py`，导出所有模块。

---

## Step 8：端到端测试 & 调参

### 你要干什么

把整个流水线跑通：从"目标位姿"到"关节轨迹"，验证每一步都正确。

### 为什么这么做

- 单元测试只能验证单个模块，端到端测试验证整个流水线
- 比赛现场调试时间有限，提前测试好能节省大量时间

### 测试清单

1. **坐标转换测试**：给定已知的相机坐标，验证转换后的基座坐标是否正确
2. **IK 求解测试**：给定已知位姿，验证 IK 解是否正确（用 FK 反算验证）
3. **轨迹规划测试**：给定起始和目标，验证轨迹是否无碰撞、平滑
4. **状态机测试**：跑一遍完整的抓取流程
5. **集成测试**：模拟 policy_server 调用，验证接口返回格式

### 发给你的子 Prompt

> **【子 Prompt - Step 8 端到端测试】**
> 请帮我创建运动规划模块的端到端测试脚本。
>
> 要求：
> 1. 在 `x-trainer/source/leisaac/leisaac/motion_planning/` 目录下创建 `test_motion_planning.py`
> 2. 包含以下测试用例：
>    - test_coordinate_transform：测试坐标转换精度
>    - test_ik_solver：测试 IK 求解（给定位姿 → 求解 → FK 验证）
>    - test_motion_planner：测试轨迹规划（给定起止点 → 规划 → 检查平滑性）
>    - test_state_machine：测试状态机流转
>    - test_full_pipeline：完整流水线测试（目标位姿 → 关节轨迹）
> 3. 每个测试用例：
>    - 有清晰的输入和期望输出
>    - 打印详细的测试结果（包括误差、耗时等）
>    - 用 assert 验证正确性
> 4. 包含一个 `run_all_tests()` 函数，一键运行所有测试
> 5. 所有代码用中文注释
>
> 运行方式：`python -m leisaac.motion_planning.test_motion_planning`

---

## 附录 A：完整文件清单

开发完成后，你的 `motion_planning/` 目录应该是这样的：

```
x-trainer/source/leisaac/leisaac/motion_planning/
├── __init__.py                 # 模块导出
├── coordinate_transform.py     # Step 3: 坐标转换
├── task_state_machine.py       # Step 4: 状态机 + 中间点
├── ik_solver.py                # Step 5: IK 求解
├── motion_planner.py           # Step 6: 避障 + 轨迹规划
├── planner_interface.py        # Step 7: 统一接口
└── test_motion_planning.py     # Step 8: 测试脚本

curobo/my_x_trainer/
├── xtrainer.urdf               # Step 1: 机器人 URDF
├── xtrainer.yml                # Step 2: cuRobo 配置文件
└── build_model.py              # Step 2: 模型构建脚本
```

## 附录 B：常见问题 FAQ

**Q1: cuRobo 报错 "CUDA not available"**
A: 确认你有 NVIDIA GPU 且安装了 CUDA 版本的 PyTorch。运行 `python -c "import torch; print(torch.cuda.is_available())"` 检查。

**Q2: IK 求解总是失败**
A: 可能原因：(1) 目标超出工作空间，尝试更近的目标；(2) num_seeds 太少，增加到 128；(3) 关节限位太紧，检查 URDF。

**Q3: 轨迹规划报碰撞**
A: 检查场景中的障碍物定义是否正确。可以先不加障碍物测试，确认基本规划能通过。

**Q4: 四元数顺序搞混了**
A: cuRobo 用 **wxyz** 顺序。如果你的数据是 xyzw，需要转换：`wxyz = [xyzw[3], xyzw[0], xyzw[1], xyzw[2]]`

**Q5: 工作空间路径中有中文字符导致报错**
A: 设置环境变量 `PYTHONUTF8=1`，或者把项目移到纯英文路径下。

---

## 附录 C：关键 API 速查表

| 功能 | import 路径 | 关键方法 |
|------|------------|---------|
| 正运动学 | `curobo.kinematics` | `Kinematics.compute_kinematics(js)` |
| 逆运动学 | `curobo.inverse_kinematics` | `InverseKinematics.solve_pose(pose)` |
| 运动规划 | `curobo.motion_planner` | `MotionPlanner.plan_pose(goal, start)` |
| 轨迹优化 | `curobo.trajectory_optimizer` | `TrajectoryOptimizer.solve_pose(pose)` |
| 场景管理 | `curobo.scene` | `Scene(cuboid=[...], sphere=[...])` |
| 关节状态 | `curobo.types` | `JointState.from_position(q)` |
| 位姿 | `curobo.types` | `Pose(position, quaternion)` |
| 碰撞检测 | `curobo.collision_checking` | `CollisionChecker.check(js)` |
| 机器人构建 | `curobo.robot_builder` | `RobotBuilder(urdf, asset_path)` |
