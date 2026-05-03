"""
双臂抓取任务状态机 + 路径中间点管理模块
==========================================

功能：
  管理双臂抓取任务的完整流程，将复杂的抓取动作分解为可管理的状态。

状态流转：
  IDLE → DETECT → MOVE_ABOVE → DESCEND → GRASP → LIFT → TRANSIT
  → PLACE_DESCEND → RELEASE → RETREAT → DONE

设计思路：
  - 每个状态有明确的目标、进入条件、超时处理、失败回退
  - 左右臂独立运行各自的状态机，互不干扰
  - WaypointManager 负责生成和管理路径中间点

依赖：仅 numpy + 标准库
"""

import time
import numpy as np
from typing import Optional, List, Dict, Callable
from dataclasses import dataclass, field
from enum import Enum, auto


# ============================================================
# 第一部分：数据结构定义
# ============================================================

class TaskState(Enum):
    """
    任务状态枚举。

    每个状态对应抓取流程中的一个阶段。
    状态机会按顺序在这些状态之间流转。
    """
    IDLE = auto()             # 空闲，等待任务
    DETECT = auto()           # 检测目标位置
    MOVE_ABOVE = auto()       # 移动到目标上方（安全高度）
    DESCEND = auto()          # 垂直下降到抓取位置
    GRASP = auto()            # 闭合夹爪
    LIFT = auto()             # 垂直提升
    TRANSIT = auto()          # 移动到放置区上方
    PLACE_DESCEND = auto()    # 下降到放置位置
    RELEASE = auto()          # 打开夹爪
    RETREAT = auto()          # 后撤到安全位置
    DONE = auto()             # 任务完成
    ERROR = auto()            # 出错，需要处理


# 状态之间的合法转移路径
# 格式：{当前状态: [可能的下一状态]}
VALID_TRANSITIONS = {
    TaskState.IDLE:          [TaskState.DETECT],
    TaskState.DETECT:        [TaskState.MOVE_ABOVE, TaskState.ERROR],
    TaskState.MOVE_ABOVE:    [TaskState.DESCEND, TaskState.ERROR],
    TaskState.DESCEND:       [TaskState.GRASP, TaskState.ERROR],
    TaskState.GRASP:         [TaskState.LIFT, TaskState.ERROR],
    TaskState.LIFT:          [TaskState.TRANSIT, TaskState.ERROR],
    TaskState.TRANSIT:       [TaskState.PLACE_DESCEND, TaskState.ERROR],
    TaskState.PLACE_DESCEND: [TaskState.RELEASE, TaskState.ERROR],
    TaskState.RELEASE:       [TaskState.RETREAT, TaskState.ERROR],
    TaskState.RETREAT:       [TaskState.DONE, TaskState.ERROR],
    TaskState.DONE:          [TaskState.IDLE],       # 完成后可以重新开始
    TaskState.ERROR:         [TaskState.IDLE, TaskState.MOVE_ABOVE],  # 错误后可重试
}


@dataclass
class Pose:
    """
    位姿数据类。

    表示一个物体在 3D 空间中的位置和姿态。
    使用四元数表示旋转（cuRobo 约定：wxyz 顺序）。
    """
    position: np.ndarray      # [x, y, z] 位置（米）
    quaternion: np.ndarray     # [qw, qx, qy, qz] 姿态四元数

    def __post_init__(self):
        """确保输入是 numpy 数组"""
        self.position = np.asarray(self.position, dtype=np.float64)
        self.quaternion = np.asarray(self.quaternion, dtype=np.float64)

    def copy(self):
        """深拷贝"""
        return Pose(self.position.copy(), self.quaternion.copy())

    def __repr__(self):
        p = self.position
        q = self.quaternion
        return f"Pose(pos=[{p[0]:.3f},{p[1]:.3f},{p[2]:.3f}], quat=[{q[0]:.3f},{q[1]:.3f},{q[2]:.3f},{q[3]:.3f}])"


@dataclass
class Waypoint:
    """
    路径中间点数据类。

    每个中间点包含：
      - 目标位姿
      - 夹爪状态（0.0=闭合，1.0=张开）
      - 速度系数（0~1，控制运动快慢）
      - 中文描述（便于调试）
    """
    pose: Pose
    gripper_state: float = 0.0       # 0.0=闭合, 1.0=张开
    speed_factor: float = 0.5        # 速度系数 0~1
    description: str = ""            # 中文描述

    def __repr__(self):
        return f"Waypoint({self.description}, gripper={self.gripper_state}, speed={self.speed_factor})"


@dataclass
class TaskConfig:
    """
    任务配置参数。

    包含抓取任务中的各种偏移量和阈值，可以通过修改这些参数来调整行为。
    """
    approach_offset_z: float = 0.10      # 抓取前的安全高度偏移（米）
    lift_offset_z: float = 0.15          # 抓取后的提升高度偏移（米）
    retreat_offset_z: float = 0.10       # 后撤安全高度偏移（米）
    gripper_open: float = 1.0            # 夹爪张开状态值
    gripper_close: float = 0.0           # 夹爪闭合状态值
    grasp_speed: float = 0.3             # 接近/下降速度系数
    transit_speed: float = 0.6           # 移动速度系数
    state_timeout: float = 10.0          # 单个状态的超时时间（秒）
    grasp_settle_time: float = 0.5       # 抓取后等待稳定的时间（秒）
    release_settle_time: float = 0.3     # 释放后等待的时间（秒）


# ============================================================
# 第二部分：中间点管理器
# ============================================================

class WaypointManager:
    """
    路径中间点管理器。

    功能：
      1. 管理一个有序的中间点队列
      2. 自动生成抓取路径的中间点序列
      3. 提供按顺序获取中间点的接口

    使用流程：
      >>> manager = WaypointManager()
      >>> manager.generate_grasp_waypoints(target_pose, place_pose)
      >>> while manager.has_next():
      ...     wp = manager.get_next_waypoint()
      ...     # 执行这个中间点的运动
    """

    def __init__(self):
        """初始化空的中间点队列"""
        self._waypoints: List[Waypoint] = []
        self._index: int = 0    # 当前读取位置

    def clear(self):
        """清空所有中间点"""
        self._waypoints.clear()
        self._index = 0

    def add_waypoint(self, pose: Pose, gripper_state: float = 0.0,
                     speed_factor: float = 0.5, description: str = ""):
        """
        添加一个中间点到队列末尾。

        参数：
            pose: 目标位姿
            gripper_state: 夹爪状态（0.0=闭合，1.0=张开）
            speed_factor: 速度系数（0~1）
            description: 中文描述
        """
        wp = Waypoint(
            pose=pose.copy(),
            gripper_state=gripper_state,
            speed_factor=speed_factor,
            description=description,
        )
        self._waypoints.append(wp)

    def has_next(self) -> bool:
        """检查是否还有未处理的中间点"""
        return self._index < len(self._waypoints)

    def get_next_waypoint(self) -> Optional[Waypoint]:
        """
        获取下一个中间点并推进索引。

        返回：
            下一个 Waypoint，如果没有则返回 None
        """
        if not self.has_next():
            return None
        wp = self._waypoints[self._index]
        self._index += 1
        return wp

    def peek_next(self) -> Optional[Waypoint]:
        """
        查看下一个中间点但不推进索引（用于预览）。

        返回：
            下一个 Waypoint，如果没有则返回 None
        """
        if not self.has_next():
            return None
        return self._waypoints[self._index]

    def reset(self):
        """重置索引到开头（可以重新遍历中间点）"""
        self._index = 0

    @property
    def current_index(self) -> int:
        """当前已处理到第几个中间点"""
        return self._index

    @property
    def total_count(self) -> int:
        """中间点总数"""
        return len(self._waypoints)

    @property
    def remaining_count(self) -> int:
        """剩余未处理的中间点数量"""
        return max(0, len(self._waypoints) - self._index)

    def get_all_waypoints(self) -> List[Waypoint]:
        """获取所有中间点的列表"""
        return list(self._waypoints)

    # ---- 自动生成抓取路径中间点 ----

    def generate_grasp_waypoints(self, grasp_pose: Pose, place_pose: Pose,
                                  config: Optional[TaskConfig] = None):
        """
        自动生成完整的抓取-放置路径中间点序列。

        生成的路径：
          1. 移动到抓取目标上方（安全高度）
          2. 下降到抓取位置
          3. 闭合夹爪
          4. 提升到安全高度
          5. 移动到放置区上方
          6. 下降到放置位置
          7. 张开夹爪
          8. 后撤到安全高度

        参数：
            grasp_pose: 抓取目标位姿（物体位置）
            place_pose: 放置目标位姿（放置位置）
            config: 任务配置参数（偏移量、速度等）
        """
        if config is None:
            config = TaskConfig()

        self.clear()

        # ---- 阶段 1：接近 ----
        # 移动到抓取目标正上方，z 方向抬高 approach_offset_z
        above_grasp = grasp_pose.copy()
        above_grasp.position[2] += config.approach_offset_z
        self.add_waypoint(
            pose=above_grasp,
            gripper_state=config.gripper_open,   # 夹爪张开
            speed_factor=config.transit_speed,
            description="移动到抓取目标上方",
        )

        # ---- 阶段 2：下降 ----
        # 垂直下降到抓取位置
        self.add_waypoint(
            pose=grasp_pose.copy(),
            gripper_state=config.gripper_open,   # 保持张开
            speed_factor=config.grasp_speed,      # 慢速下降
            description="下降到抓取位置",
        )

        # ---- 阶段 3：抓取 ----
        # 闭合夹爪
        self.add_waypoint(
            pose=grasp_pose.copy(),
            gripper_state=config.gripper_close,  # 闭合夹爪
            speed_factor=config.grasp_speed,
            description="闭合夹爪",
        )

        # ---- 阶段 4：提升 ----
        # 垂直提升到安全高度
        above_grasp_lift = grasp_pose.copy()
        above_grasp_lift.position[2] += config.lift_offset_z
        self.add_waypoint(
            pose=above_grasp_lift,
            gripper_state=config.gripper_close,  # 保持闭合
            speed_factor=config.grasp_speed,
            description="提升到安全高度",
        )

        # ---- 阶段 5：转运 ----
        # 移动到放置区上方
        above_place = place_pose.copy()
        above_place.position[2] += config.approach_offset_z
        self.add_waypoint(
            pose=above_place,
            gripper_state=config.gripper_close,  # 保持闭合
            speed_factor=config.transit_speed,
            description="移动到放置区上方",
        )

        # ---- 阶段 6：放置下降 ----
        # 下降到放置位置
        self.add_waypoint(
            pose=place_pose.copy(),
            gripper_state=config.gripper_close,  # 保持闭合
            speed_factor=config.grasp_speed,
            description="下降到放置位置",
        )

        # ---- 阶段 7：释放 ----
        # 张开夹爪
        self.add_waypoint(
            pose=place_pose.copy(),
            gripper_state=config.gripper_open,   # 张开夹爪
            speed_factor=config.grasp_speed,
            description="张开夹爪释放物体",
        )

        # ---- 阶段 8：后撤 ----
        # 后撤到安全高度
        above_place_retreat = place_pose.copy()
        above_place_retreat.position[2] += config.retreat_offset_z
        self.add_waypoint(
            pose=above_place_retreat,
            gripper_state=config.gripper_open,   # 保持张开
            speed_factor=config.transit_speed,
            description="后撤到安全位置",
        )

    def __repr__(self):
        return (f"WaypointManager(total={self.total_count}, "
                f"done={self.current_index}, remaining={self.remaining_count})")


# ============================================================
# 第三部分：状态机核心
# ============================================================

class GraspTaskStateMachine:
    """
    双臂抓取任务状态机。

    管理单臂的抓取任务流程。双臂使用两个独立的状态机实例。

    使用流程：
      >>> config = TaskConfig()
      >>> fsm = GraspTaskStateMachine("left", config)
      >>> fsm.start(grasp_pose, place_pose)
      >>> while not fsm.is_done():
      ...     fsm.update()                    # 更新状态
      ...     target = fsm.get_target_pose()  # 获取当前目标
      ...     gripper = fsm.get_gripper_cmd()  # 获取夹爪命令
      ...     # 执行运动...

    状态流转：
      IDLE → DETECT → MOVE_ABOVE → DESCEND → GRASP → LIFT
      → TRANSIT → PLACE_DESCEND → RELEASE → RETREAT → DONE

    错误处理：
      任何状态超时或失败 → ERROR → 根据策略回退或重试
    """

    def __init__(self, arm: str = "left", config: Optional[TaskConfig] = None):
        """
        参数：
            arm: "left" 或 "right"，标识控制哪只手臂
            config: 任务配置参数
        """
        assert arm in ("left", "right"), f"arm 必须是 'left' 或 'right'，收到 '{arm}'"
        self.arm = arm
        self.config = config or TaskConfig()

        # 当前状态
        self._state = TaskState.IDLE
        self._previous_state = TaskState.IDLE

        # 中间点管理器
        self._waypoint_manager = WaypointManager()

        # 当前目标
        self._target_pose: Optional[Pose] = None
        self._gripper_command: float = self.config.gripper_open

        # 任务参数
        self._grasp_pose: Optional[Pose] = None
        self._place_pose: Optional[Pose] = None

        # 时间跟踪
        self._state_entry_time: float = 0.0
        self._task_start_time: float = 0.0

        # 错误信息
        self._error_msg: str = ""

        # 状态历史（用于调试）
        self._history: List[Dict] = []

    # ---- 状态查询 ----

    @property
    def state(self) -> TaskState:
        """当前状态"""
        return self._state

    @property
    def state_name(self) -> str:
        """当前状态的中文名称"""
        state_names = {
            TaskState.IDLE: "空闲",
            TaskState.DETECT: "检测目标",
            TaskState.MOVE_ABOVE: "移动到目标上方",
            TaskState.DESCEND: "下降到抓取位置",
            TaskState.GRASP: "闭合夹爪",
            TaskState.LIFT: "提升",
            TaskState.TRANSIT: "转运到放置区",
            TaskState.PLACE_DESCEND: "下降到放置位置",
            TaskState.RELEASE: "释放物体",
            TaskState.RETREAT: "后撤",
            TaskState.DONE: "完成",
            TaskState.ERROR: "错误",
        }
        return state_names.get(self._state, "未知")

    def is_done(self) -> bool:
        """任务是否已完成（到达 DONE 状态）"""
        return self._state == TaskState.DONE

    def is_error(self) -> bool:
        """是否处于错误状态"""
        return self._state == TaskState.ERROR

    def is_idle(self) -> bool:
        """是否处于空闲状态"""
        return self._state == TaskState.IDLE

    def get_target_pose(self) -> Optional[Pose]:
        """获取当前目标位姿"""
        return self._target_pose

    def get_gripper_command(self) -> float:
        """获取当前夹爪命令（0.0=闭合，1.0=张开）"""
        return self._gripper_command

    def get_error_message(self) -> str:
        """获取错误信息"""
        return self._error_msg

    def get_state_elapsed_time(self) -> float:
        """获取当前状态已持续的时间（秒）"""
        return time.time() - self._state_entry_time

    def get_history(self) -> List[Dict]:
        """获取状态转换历史"""
        return list(self._history)

    # ---- 控制接口 ----

    def start(self, grasp_pose: Pose, place_pose: Pose):
        """
        启动抓取任务。

        参数：
            grasp_pose: 抓取目标位姿
            place_pose: 放置目标位姿
        """
        self._grasp_pose = grasp_pose.copy()
        self._place_pose = place_pose.copy()
        self._error_msg = ""
        self._task_start_time = time.time()

        # 生成中间点
        self._waypoint_manager.generate_grasp_waypoints(
            grasp_pose, place_pose, self.config
        )

        # 从 IDLE 转到 DETECT
        self._transition_to(TaskState.DETECT)

    def update(self):
        """
        更新状态机。

        每次调用检查当前状态是否满足转换条件，如果满足则切换到下一状态。
        应该在主循环中以固定频率调用（如 10Hz）。
        """
        if self._state in (TaskState.IDLE, TaskState.DONE):
            return  # 空闲和完成状态不需要更新

        if self._state == TaskState.ERROR:
            return  # 错误状态等待外部处理

        # 检查超时
        if self.get_state_elapsed_time() > self.config.state_timeout:
            self._handle_timeout()
            return

        # 根据当前状态执行逻辑
        if self._state == TaskState.DETECT:
            self._update_detect()
        elif self._state == TaskState.MOVE_ABOVE:
            self._update_move_above()
        elif self._state == TaskState.DESCEND:
            self._update_descend()
        elif self._state == TaskState.GRASP:
            self._update_grasp()
        elif self._state == TaskState.LIFT:
            self._update_lift()
        elif self._state == TaskState.TRANSIT:
            self._update_transit()
        elif self._state == TaskState.PLACE_DESCEND:
            self._update_place_descend()
        elif self._state == TaskState.RELEASE:
            self._update_release()
        elif self._state == TaskState.RETREAT:
            self._update_retreat()

    def signal_arrived(self):
        """
        信号：当前目标位姿已到达。

        由外部控制器调用，告诉状态机"我已经移动到目标位置了"。
        收到此信号后，状态机切换到下一个状态。
        """
        if self._state == TaskState.MOVE_ABOVE:
            self._transition_to(TaskState.DESCEND)
        elif self._state == TaskState.DESCEND:
            self._transition_to(TaskState.GRASP)
        elif self._state == TaskState.LIFT:
            self._transition_to(TaskState.TRANSIT)
        elif self._state == TaskState.TRANSIT:
            self._transition_to(TaskState.PLACE_DESCEND)
        elif self._state == TaskState.PLACE_DESCEND:
            self._transition_to(TaskState.RELEASE)
        elif self._state == TaskState.RETREAT:
            self._transition_to(TaskState.DONE)

    def signal_error(self, message: str):
        """
        信号：发生错误。

        参数：
            message: 错误描述
        """
        self._error_msg = message
        self._transition_to(TaskState.ERROR)

    def retry(self):
        """
        从错误状态重试。

        回退到 MOVE_ABOVE 状态重新开始（跳过 DETECT，因为目标位置已知）。
        """
        if self._state == TaskState.ERROR:
            # 重新生成中间点
            if self._grasp_pose is not None and self._place_pose is not None:
                self._waypoint_manager.generate_grasp_waypoints(
                    self._grasp_pose, self._place_pose, self.config
                )
            self._error_msg = ""
            self._transition_to(TaskState.MOVE_ABOVE)

    def reset(self):
        """重置状态机到初始状态"""
        self._state = TaskState.IDLE
        self._previous_state = TaskState.IDLE
        self._target_pose = None
        self._gripper_command = self.config.gripper_open
        self._grasp_pose = None
        self._place_pose = None
        self._error_msg = ""
        self._waypoint_manager.clear()
        self._history.clear()

    # ---- 内部方法 ----

    def _transition_to(self, new_state: TaskState):
        """
        状态转换（内部方法）。

        记录转换历史，更新时间戳。
        """
        old_state = self._state
        self._previous_state = old_state
        self._state = new_state
        self._state_entry_time = time.time()

        # 记录历史
        self._history.append({
            "from": old_state.name,
            "to": new_state.name,
            "time": time.time(),
        })

        # 进入新状态时的初始化
        self._on_enter_state(new_state)

    def _on_enter_state(self, state: TaskState):
        """
        进入新状态时的初始化逻辑。

        根据进入的状态，设置目标位姿和夹爪命令。
        """
        if state == TaskState.DETECT:
            # 检测阶段：获取下一个中间点
            self._advance_waypoint()

        elif state == TaskState.MOVE_ABOVE:
            # 目标已经在 DETECT 阶段设置
            pass

        elif state == TaskState.DESCEND:
            self._advance_waypoint()

        elif state == TaskState.GRASP:
            # 闭合夹爪
            self._gripper_command = self.config.gripper_close
            self._advance_waypoint()

        elif state == TaskState.LIFT:
            self._advance_waypoint()

        elif state == TaskState.TRANSIT:
            self._advance_waypoint()

        elif state == TaskState.PLACE_DESCEND:
            self._advance_waypoint()

        elif state == TaskState.RELEASE:
            # 张开夹爪
            self._gripper_command = self.config.gripper_open
            self._advance_waypoint()

        elif state == TaskState.RETREAT:
            self._advance_waypoint()

    def _advance_waypoint(self):
        """
        从中间点管理器获取下一个中间点，更新目标位姿和夹爪状态。
        """
        wp = self._waypoint_manager.get_next_waypoint()
        if wp is not None:
            self._target_pose = wp.pose
            self._gripper_command = wp.gripper_state

    def _handle_timeout(self):
        """
        处理状态超时。

        超时后进入 ERROR 状态。
        """
        self._error_msg = f"状态 {self.state_name} 超时（{self.config.state_timeout}秒）"
        self._transition_to(TaskState.ERROR)

    # ---- 各状态的更新逻辑 ----

    def _update_detect(self):
        """
        DETECT 状态更新。

        在实际应用中，这里会调用视觉检测模块。
        由于检测结果通过 start() 的 grasp_pose 参数已经确定，
        这里直接转到 MOVE_ABOVE。
        """
        # 如果已有抓取目标，直接进入 MOVE_ABOVE
        if self._grasp_pose is not None:
            self._transition_to(TaskState.MOVE_ABOVE)

    def _update_move_above(self):
        """
        MOVE_ABOVE 状态更新。

        目标是移动到抓取目标上方。
        等待外部调用 signal_arrived() 表示到达。
        """
        pass  # 等待 signal_arrived()

    def _update_descend(self):
        """
        DESCEND 状态更新。

        目标是垂直下降到抓取位置。
        等待外部调用 signal_arrived() 表示到达。
        """
        pass  # 等待 signal_arrived()

    def _update_grasp(self):
        """
        GRASP 状态更新。

        闭合夹爪后，等待一段时间让夹爪稳定。
        然后自动转到 LIFT。
        """
        if self.get_state_elapsed_time() > self.config.grasp_settle_time:
            self._transition_to(TaskState.LIFT)

    def _update_lift(self):
        """
        LIFT 状态更新。

        目标是垂直提升。
        等待外部调用 signal_arrived() 表示到达。
        """
        pass  # 等待 signal_arrived()

    def _update_transit(self):
        """
        TRANSIT 状态更新。

        目标是移动到放置区上方。
        等待外部调用 signal_arrived() 表示到达。
        """
        pass  # 等待 signal_arrived()

    def _update_place_descend(self):
        """
        PLACE_DESCEND 状态更新。

        目标是下降到放置位置。
        等待外部调用 signal_arrived() 表示到达。
        """
        pass  # 等待 signal_arrived()

    def _update_release(self):
        """
        RELEASE 状态更新。

        张开夹爪后，等待一段时间。
        然后自动转到 RETREAT。
        """
        if self.get_state_elapsed_time() > self.config.release_settle_time:
            self._transition_to(TaskState.RETREAT)

    def _update_retreat(self):
        """
        RETREAT 状态更新。

        目标是后撤到安全位置。
        等待外部调用 signal_arrived() 表示到达。
        """
        pass  # 等待 signal_arrived()

    def __repr__(self):
        return (f"GraspTaskStateMachine(arm={self.arm}, state={self.state_name}, "
                f"waypoints={self._waypoint_manager})")


# ============================================================
# 第四部分：双臂协调器
# ============================================================

class DualArmCoordinator:
    """
    双臂协调器。

    管理左右臂的两个独立状态机，提供统一的控制接口。
    左右臂可以处于不同的状态，独立执行各自的抓取任务。

    使用流程：
      >>> coordinator = DualArmCoordinator()
      >>> coordinator.start_left(left_grasp_pose, left_place_pose)
      >>> coordinator.start_right(right_grasp_pose, right_place_pose)
      >>> while not coordinator.is_all_done():
      ...     coordinator.update_all()
      ...     left_target = coordinator.get_target("left")
      ...     right_target = coordinator.get_target("right")
    """

    def __init__(self, config: Optional[TaskConfig] = None):
        """
        参数：
            config: 任务配置参数（左右臂共享）
        """
        self.config = config or TaskConfig()
        self._left_fsm = GraspTaskStateMachine("left", self.config)
        self._right_fsm = GraspTaskStateMachine("right", self.config)

    @property
    def left(self) -> GraspTaskStateMachine:
        """左臂状态机"""
        return self._left_fsm

    @property
    def right(self) -> GraspTaskStateMachine:
        """右臂状态机"""
        return self._right_fsm

    def start_left(self, grasp_pose: Pose, place_pose: Pose):
        """启动左臂抓取任务"""
        self._left_fsm.start(grasp_pose, place_pose)

    def start_right(self, grasp_pose: Pose, place_pose: Pose):
        """启动右臂抓取任务"""
        self._right_fsm.start(grasp_pose, place_pose)

    def start_both(self, left_grasp: Pose, left_place: Pose,
                   right_grasp: Pose, right_place: Pose):
        """同时启动双臂抓取任务"""
        self.start_left(left_grasp, left_place)
        self.start_right(right_grasp, right_place)

    def update_all(self):
        """更新所有状态机"""
        self._left_fsm.update()
        self._right_fsm.update()

    def get_target(self, arm: str) -> Optional[Pose]:
        """
        获取指定手臂的当前目标位姿。

        参数：
            arm: "left" 或 "right"

        返回：
            目标位姿，如果没有则返回 None
        """
        if arm == "left":
            return self._left_fsm.get_target_pose()
        elif arm == "right":
            return self._right_fsm.get_target_pose()
        else:
            raise ValueError(f"arm 必须是 'left' 或 'right'，收到 '{arm}'")

    def get_gripper(self, arm: str) -> float:
        """
        获取指定手臂的夹爪命令。

        参数：
            arm: "left" 或 "right"

        返回：
            夹爪状态（0.0=闭合，1.0=张开）
        """
        if arm == "left":
            return self._left_fsm.get_gripper_command()
        elif arm == "right":
            return self._right_fsm.get_gripper_command()
        else:
            raise ValueError(f"arm 必须是 'left' 或 'right'，收到 '{arm}'")

    def is_all_done(self) -> bool:
        """双臂是否都已完成"""
        return self._left_fsm.is_done() and self._right_fsm.is_done()

    def has_error(self) -> bool:
        """是否有手臂处于错误状态"""
        return self._left_fsm.is_error() or self._right_fsm.is_error()

    def get_status(self) -> Dict:
        """
        获取双臂状态摘要。

        返回：
            包含左右臂状态信息的字典
        """
        return {
            "left": {
                "state": self._left_fsm.state_name,
                "is_done": self._left_fsm.is_done(),
                "is_error": self._left_fsm.is_error(),
                "error": self._left_fsm.get_error_message(),
                "waypoints_remaining": self._left_fsm._waypoint_manager.remaining_count,
            },
            "right": {
                "state": self._right_fsm.state_name,
                "is_done": self._right_fsm.is_done(),
                "is_error": self._right_fsm.is_error(),
                "error": self._right_fsm.get_error_message(),
                "waypoints_remaining": self._right_fsm._waypoint_manager.remaining_count,
            },
        }

    def reset_all(self):
        """重置所有状态机"""
        self._left_fsm.reset()
        self._right_fsm.reset()

    def __repr__(self):
        return (f"DualArmCoordinator(left={self._left_fsm.state_name}, "
                f"right={self._right_fsm.state_name})")


# ============================================================
# 第五部分：测试函数
# ============================================================

def test_state_machine():
    """
    测试状态机和中间点管理器。

    测试内容：
      1. 中间点生成
      2. 单臂状态机完整流程
      3. 双臂协调器
      4. 错误处理和重试
    """
    print("=" * 60)
    print("状态机 + 中间点管理器 测试")
    print("=" * 60)

    all_pass = True

    # ---- 测试 1：中间点生成 ----
    print("\n[测试 1] 中间点生成")
    wm = WaypointManager()
    grasp_pose = Pose(
        position=np.array([0.3, 0.0, 0.1]),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    place_pose = Pose(
        position=np.array([0.3, 0.3, 0.1]),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    wm.generate_grasp_waypoints(grasp_pose, place_pose)
    passed = wm.total_count == 8
    all_pass = all_pass and passed
    print(f"  中间点数量: {wm.total_count} (期望: 8)")
    print(f"  结果: {'PASS' if passed else 'FAIL'}")

    for i, wp in enumerate(wm.get_all_waypoints()):
        print(f"    [{i}] {wp.description} | gripper={wp.gripper_state} speed={wp.speed_factor}")

    # ---- 测试 2：单臂状态机完整流程 ----
    print("\n[测试 2] 单臂状态机完整流程")
    config = TaskConfig(state_timeout=5.0, grasp_settle_time=0.1, release_settle_time=0.1)
    fsm = GraspTaskStateMachine("left", config)
    print(f"  初始状态: {fsm.state_name}")
    passed = fsm.is_idle()
    all_pass = all_pass and passed
    print(f"  初始为 IDLE: {'PASS' if passed else 'FAIL'}")

    # 启动任务
    fsm.start(grasp_pose, place_pose)
    print(f"  启动后状态: {fsm.state_name}")

    # 模拟执行流程
    max_steps = 30
    step = 0
    while not fsm.is_done() and not fsm.is_error() and step < max_steps:
        fsm.update()

        # 对于需要等待到达的状态，模拟到达信号
        if fsm.state in (TaskState.MOVE_ABOVE, TaskState.DESCEND,
                         TaskState.LIFT, TaskState.TRANSIT,
                         TaskState.PLACE_DESCEND, TaskState.RETREAT):
            fsm.signal_arrived()

        step += 1
        time.sleep(0.02)  # 让时间流逝，使 settle_time 生效

    passed = fsm.is_done()
    all_pass = all_pass and passed
    print(f"  最终状态: {fsm.state_name}")
    print(f"  完成任务: {'PASS' if passed else 'FAIL'}")
    print(f"  状态转换次数: {len(fsm.get_history())}")
    print(f"  执行步骤数: {step}")

    # ---- 测试 3：双臂协调器 ----
    print("\n[测试 3] 双臂协调器")
    coordinator = DualArmCoordinator(config)

    left_grasp = Pose(np.array([0.3, -0.2, 0.1]), np.array([1, 0, 0, 0]))
    left_place = Pose(np.array([0.3, -0.2, 0.3]), np.array([1, 0, 0, 0]))
    right_grasp = Pose(np.array([0.3, 0.2, 0.1]), np.array([1, 0, 0, 0]))
    right_place = Pose(np.array([0.3, 0.2, 0.3]), np.array([1, 0, 0, 0]))

    coordinator.start_both(left_grasp, left_place, right_grasp, right_place)
    print(f"  启动后: {coordinator}")

    # 模拟执行
    for _ in range(30):
        coordinator.update_all()
        for arm in ("left", "right"):
            fsm = coordinator.left if arm == "left" else coordinator.right
            if fsm.state in (TaskState.MOVE_ABOVE, TaskState.DESCEND,
                             TaskState.LIFT, TaskState.TRANSIT,
                             TaskState.PLACE_DESCEND, TaskState.RETREAT):
                fsm.signal_arrived()
        time.sleep(0.02)  # 让时间流逝，使 settle_time 生效

    passed = coordinator.is_all_done()
    all_pass = all_pass and passed
    print(f"  最终状态: {coordinator}")
    print(f"  双臂完成: {'PASS' if passed else 'FAIL'}")
    print(f"  状态摘要: {coordinator.get_status()}")

    # ---- 测试 4：错误处理 ----
    print("\n[测试 4] 错误处理和重试")
    fsm_err = GraspTaskStateMachine("right", config)
    fsm_err.start(grasp_pose, place_pose)
    fsm_err.signal_error("IK 求解失败")

    passed = fsm_err.is_error()
    all_pass = all_pass and passed
    print(f"  触发错误后状态: {fsm_err.state_name}")
    print(f"  错误标志: {'PASS' if passed else 'FAIL'}")
    print(f"  错误信息: {fsm_err.get_error_message()}")

    # 重试
    fsm_err.retry()
    passed = fsm_err.state == TaskState.MOVE_ABOVE
    all_pass = all_pass and passed
    print(f"  重试后状态: {fsm_err.state_name}")
    print(f"  重试成功: {'PASS' if passed else 'FAIL'}")

    # ---- 测试 5：WaypointManager 边界情况 ----
    print("\n[测试 5] WaypointManager 边界情况")
    wm_empty = WaypointManager()
    passed = not wm_empty.has_next() and wm_empty.get_next_waypoint() is None
    all_pass = all_pass and passed
    print(f"  空管理器: {'PASS' if passed else 'FAIL'}")

    wm_single = WaypointManager()
    wm_single.add_waypoint(grasp_pose, description="单个点")
    passed = wm_single.total_count == 1 and wm_single.has_next()
    all_pass = all_pass and passed
    print(f"  单点管理器: {'PASS' if passed else 'FAIL'}")

    wp = wm_single.get_next_waypoint()
    passed = wp is not None and wp.description == "单个点" and not wm_single.has_next()
    all_pass = all_pass and passed
    print(f"  获取后为空: {'PASS' if passed else 'FAIL'}")

    # ---- 总结 ----
    print("\n" + "=" * 60)
    if all_pass:
        print("全部测试通过！")
    else:
        print("部分测试失败，请检查代码")
    print("=" * 60)

    return all_pass


# ============================================================
# 主函数入口
# ============================================================

if __name__ == "__main__":
    test_state_machine()
