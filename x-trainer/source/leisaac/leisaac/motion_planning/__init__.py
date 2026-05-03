"""
motion_planning —— Dobot X-Trainer 双臂运动规划模块
====================================================

模块结构：
  coordinate_transform  坐标转换（相机 → 基座）
  task_state_machine    任务状态机 + 中间点管理
  ik_solver             cuRobo IK 逆运动学求解
  motion_planner        cuRobo 避障轨迹规划
  planner_interface     统一接口（供 policy_server 调用）

依赖关系：
  planner_interface 依赖以上所有子模块

用法：
  # 推荐：通过统一接口使用
  from leisaac.motion_planning import MotionPlanningModule
  module = MotionPlanningModule()
  module.warmup()
  result = module.plan(target_pose, obstacles, arm="left")

  # 也可以单独使用子模块
  from leisaac.motion_planning import CameraToBaseTransformer
  from leisaac.motion_planning import DualArmIKSolver
  from leisaac.motion_planning import DualArmMotionPlanner
"""

# ---- 始终可用的模块（无 GPU 依赖）----
from .coordinate_transform import CameraToBaseTransformer
from .task_state_machine import (
    GraspTaskStateMachine,
    WaypointManager,
    Pose,
    TaskConfig,
    TaskState,
    DualArmCoordinator,
)

# ---- 需要 curobo + torch GPU 环境的模块 ----
try:
    from .ik_solver import DualArmIKSolver
    from .motion_planner import DualArmMotionPlanner, SceneBuilder
    from .planner_interface import MotionPlanningModule, PlanResult
    _HAS_CUROBO = True
except ImportError:
    _HAS_CUROBO = False

__all__ = [
    # 始终可用
    "CameraToBaseTransformer",
    "GraspTaskStateMachine",
    "WaypointManager",
    "Pose",
    "TaskConfig",
    "TaskState",
    "DualArmCoordinator",
    # 需要 curobo
    "DualArmIKSolver",
    "DualArmMotionPlanner",
    "SceneBuilder",
    "MotionPlanningModule",
    "PlanResult",
]
