"""
运动规划统一接口模块
====================

功能：
  整合坐标转换、IK 求解、轨迹规划、任务状态机，对外提供统一的运动规划接口。
  供 policy_server（gRPC）调用，输入末端执行器目标位姿，输出双臂关节轨迹。

核心流程：
  policy_server 收到策略推理结果（末端位姿）
    → plan() 检查工作空间 → IK 求解 → 避障轨迹规划 → 返回轨迹
    → policy_server 将轨迹发送给机器人执行

依赖：coordinate_transform, ik_solver, motion_planner, task_state_machine
"""

import json
import time
import numpy as np
from typing import Dict, List, Optional

from .coordinate_transform import CameraToBaseTransformer
from .task_state_machine import (
    GraspTaskStateMachine, WaypointManager, Pose, TaskConfig, TaskState,
)

try:
    from .ik_solver import DualArmIKSolver
    from .motion_planner import DualArmMotionPlanner, SceneBuilder
    _HAS_CUROBO = True
except ImportError:
    _HAS_CUROBO = False


# ============================================================
# 第一部分：规划结果数据结构
# ============================================================

class PlanResult:
    """
    规划结果 —— 可 JSON 序列化，方便 gRPC 传输。

    字段：
        success:           是否成功
        arm:               "left" 或 "right"
        trajectory:        关节轨迹 list[list[float]]，每个元素是 6 个关节角度
        duration:          轨迹时长（秒）
        n_waypoints:       轨迹点数量
        target_joints:     目标关节角度 list[float]
        gripper_command:   夹爪命令（0.0=闭合，1.0=张开）
        phase:             当前阶段名称
        error_message:     错误信息（成功时为空字符串）
    """

    __slots__ = [
        "success", "arm", "trajectory", "duration", "n_waypoints",
        "target_joints", "gripper_command", "phase", "error_message",
    ]

    def __init__(self, success: bool, arm: str = "",
                 trajectory: Optional[List[List[float]]] = None,
                 duration: float = 0.0, n_waypoints: int = 0,
                 target_joints: Optional[List[float]] = None,
                 gripper_command: float = 1.0,
                 phase: str = "", error_message: str = ""):
        self.success = success
        self.arm = arm
        self.trajectory = trajectory
        self.duration = duration
        self.n_waypoints = n_waypoints
        self.target_joints = target_joints
        self.gripper_command = gripper_command
        self.phase = phase
        self.error_message = error_message

    # ---- JSON 序列化 ----

    def to_dict(self) -> Dict:
        """转换为字典，可直接 json.dumps 发送给 gRPC 客户端。"""
        return {
            "success": self.success,
            "arm": self.arm,
            "trajectory": self.trajectory,
            "duration": self.duration,
            "n_waypoints": self.n_waypoints,
            "target_joints": self.target_joints,
            "gripper_command": self.gripper_command,
            "phase": self.phase,
            "error_message": self.error_message,
        }

    def to_json(self) -> str:
        """序列化为 JSON 字符串。"""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Dict) -> "PlanResult":
        """从字典反序列化。"""
        return cls(**{k: d[k] for k in cls.__slots__ if k in d})

    def __repr__(self):
        status = "OK" if self.success else "FAIL"
        return (
            f"PlanResult({status}, arm={self.arm}, "
            f"wp={self.n_waypoints}, dur={self.duration:.2f}s)"
        )


# ============================================================
# 第二部分：运动规划统一接口
# ============================================================

class MotionPlanningModule:
    """
    运动规划统一接口。

    整合所有子模块，为 policy_server 提供简洁的规划 API。

    典型使用（gRPC handler 中）：
        >>> module = MotionPlanningModule()
        >>> module.warmup()
        >>>
        >>> # 收到策略推理结果后调用
        >>> result = module.plan(
        ...     target_pose={"position": [0.3, -0.2, 0.18], "quaternion": [1,0,0,0]},
        ...     obstacles=[{"name": "桌子", "position": [0.4,0,0.15], "dimensions": [0.8,0.6,0.03]}],
        ...     arm="left",
        ... )
        >>> if result.success:
        ...     send_to_robot(result.to_json())
    """

    def __init__(self, robot_config: str = "xtrainer.yml",
                 ik_num_seeds: int = 32,
                 self_collision_check: bool = True,
                 accept_converged_ik_without_feasible: bool = True):
        """
        初始化运动规划模块。

        参数：
            robot_config:        机器人配置文件（xtrainer.yml）
            ik_num_seeds:        IK 并行种子数
            self_collision_check: 是否启用自碰撞检测
        """
        if not _HAS_CUROBO:
            raise ImportError(
                "cuRobo 未安装，无法使用 MotionPlanningModule。\n"
                "请确保已安装 curobo 和 torch (CUDA 版本)。\n"
                "在 Ubuntu 上运行 setup_workstation.sh 进行部署。"
            )

        print("=" * 50)
        print("初始化运动规划模块")
        print("=" * 50)

        # ---- 子模块 1：坐标转换器 ----
        print("\n[1/3] 加载坐标转换器...")
        self._transformer = CameraToBaseTransformer()

        # ---- 子模块 2：IK 求解器 ----
        print("[2/3] 加载 IK 求解器...")
        self._ik = DualArmIKSolver(
            robot_config=robot_config,
            num_seeds=ik_num_seeds,
            self_collision_check=self_collision_check,
            accept_converged_without_feasible=accept_converged_ik_without_feasible,
        )

        # ---- 子模块 3：运动规划器 ----
        print("[3/3] 加载运动规划器...")
        self._planner = DualArmMotionPlanner(robot_config=robot_config)

        # ---- 任务配置 ----
        self._task_config = TaskConfig()

        # ---- 当前关节状态缓存（左右臂各 6 个关节）----
        self._current_joints = {
            "left": np.zeros(6, dtype=np.float32).tolist(),
            "right": np.zeros(6, dtype=np.float32).tolist(),
        }

        print("\n运动规划模块就绪！\n")

    # ================================================================
    # 公开接口 1：warmup —— 预热
    # ================================================================

    def warmup(self, num_iterations: int = 5):
        """
        预热运动规划器（编译 CUDA 图）。
        在第一次调用 plan() 之前调用，将延迟移到初始化阶段。
        """
        self._planner.warmup(num_iterations)

    # ================================================================
    # 公开接口 2：plan —— 核心规划接口
    # ================================================================

    def plan(self, target_pose: Dict,
             obstacles: Optional[List[Dict]] = None,
             arm: str = "left") -> PlanResult:
        """
        核心规划接口：从当前状态到目标位姿，生成无碰撞轨迹。

        完整流程：
          a. 参数校验 + 工作空间检查
          b. IK 求解：目标位姿 → 目标关节角度
          c. 运动规划：当前关节 → 目标关节，生成无碰撞轨迹
          d. 插值并返回密集轨迹

        参数：
            target_pose: 目标位姿
                {"position": [x, y, z], "quaternion": [qw, qx, qy, qz]}
            obstacles:  障碍物列表（可选），每个元素：
                {"name": str, "position": [x,y,z], "dimensions": [w,h,d]}
            arm:        "left" 或 "right"

        返回：
            PlanResult（可 JSON 序列化）
        """
        t0 = time.time()

        # ---- 步骤 a：参数校验 ----
        validation = self._validate(target_pose, arm)
        if validation is not None:
            return validation

        # ---- 步骤 a2：更新障碍物 ----
        if obstacles:
            self._planner.update_obstacles(obstacles)

        # ---- 步骤 b：IK 求解 ----
        ik_result = self._ik.solve_left_arm(target_pose) if arm == "left" \
            else self._ik.solve_right_arm(target_pose)

        if not ik_result["success"]:
            return PlanResult(
                success=False, arm=arm, phase="ik",
                error_message=ik_result["error_message"],
            )

        target_joints = ik_result["joint_angles"]

        # ---- 步骤 c：运动规划 ----
        current = self._current_joints[arm]
        plan_result = self._planner.plan_joint_to_joint(
            arm=arm,
            start_joints=current,
            goal_joints=target_joints,
        )

        if not plan_result["success"]:
            return PlanResult(
                success=False, arm=arm, phase="motion_plan",
                target_joints=target_joints,
                error_message=plan_result["error_message"],
            )

        # ---- 步骤 d：组装结果 ----
        trajectory = plan_result["trajectory"]
        duration = plan_result["duration"]

        # 更新内部关节状态缓存
        self._current_joints[arm] = target_joints

        elapsed = time.time() - t0
        return PlanResult(
            success=True,
            arm=arm,
            trajectory=trajectory,
            duration=duration,
            n_waypoints=len(trajectory),
            target_joints=target_joints,
            gripper_command=1.0,  # 默认张开
            phase="move",
            error_message="",
        )

    # ================================================================
    # 公开接口 3：plan_grasp_sequence —— 完整抓取序列规划
    # ================================================================

    def plan_grasp_sequence(self, target_pose: Dict,
                            place_pose: Dict,
                            obstacles: Optional[List[Dict]] = None,
                            arm: str = "left") -> PlanResult:
        """
        完整抓取序列规划：移动到抓取位 → 抓取 → 移动到放置位 → 放置。

        流程：
          1. 移动到抓取目标上方（安全高度）
          2. 下降到抓取位置
          3. 闭合夹爪
          4. 提升到安全高度
          5. 移动到放置目标上方
          6. 下降到放置位置
          7. 张开夹爪
          8. 后撤到安全位置

        每一步生成一段轨迹，最终拼接为一条完整轨迹返回。

        参数：
            target_pose: 抓取目标位姿 {"position": [x,y,z], "quaternion": [qw,qx,qy,qz]}
            place_pose:  放置目标位姿 {"position": [x,y,z], "quaternion": [qw,qx,qy,qz]}
            obstacles:   障碍物列表
            arm:         "left" 或 "right"

        返回：
            PlanResult（包含完整轨迹和各阶段信息）
        """
        t0 = time.time()

        # ---- 参数校验 ----
        for name, pose in [("抓取", target_pose), ("放置", place_pose)]:
            err = self._validate(pose, arm)
            if err is not None:
                err.phase = f"validate_{name}"
                err.error_message = f"{name}位姿校验失败：{err.error_message}"
                return err

        # ---- 更新障碍物 ----
        if obstacles:
            self._planner.update_obstacles(obstacles)

        # ---- 生成中间点 ----
        grasp_pose_obj = Pose(
            position=np.array(target_pose["position"]),
            quaternion=np.array(target_pose["quaternion"]),
        )
        place_pose_obj = Pose(
            position=np.array(place_pose["position"]),
            quaternion=np.array(place_pose["quaternion"]),
        )

        wm = WaypointManager()
        wm.generate_grasp_waypoints(grasp_pose_obj, place_pose_obj, self._task_config)

        # ---- 逐段规划 ----
        all_trajectory = []
        current_joints = self._current_joints[arm]
        waypoint_details = []

        while wm.has_next():
            wp = wm.get_next_waypoint()
            segment_pose = {
                "position": wp.pose.position.tolist(),
                "quaternion": wp.pose.quaternion.tolist(),
            }

            # IK 求解
            ik_result = self._ik.solve_left_arm(segment_pose) if arm == "left" \
                else self._ik.solve_right_arm(segment_pose)

            if not ik_result["success"]:
                return PlanResult(
                    success=False, arm=arm, phase=wp.description,
                    error_message=f"阶段 [{wp.description}] IK 失败：{ik_result['error_message']}",
                )

            # 运动规划：IK 已给出目标关节，后续使用关节空间规划，避免 cuRobo 内部重复 IK。
            plan_result = self._planner.plan_joint_to_joint(
                arm=arm,
                start_joints=current_joints,
                goal_joints=ik_result["joint_angles"],
            )

            if not plan_result["success"]:
                return PlanResult(
                    success=False, arm=arm, phase=wp.description,
                    error_message=f"阶段 [{wp.description}] 规划失败：{plan_result['error_message']}",
                )

            # 拼接轨迹
            segment_traj = plan_result["trajectory"]
            all_trajectory.extend(segment_traj)
            current_joints = ik_result["joint_angles"]

            waypoint_details.append({
                "phase": wp.description,
                "n_waypoints": len(segment_traj),
                "gripper": wp.gripper_state,
            })

        # ---- 更新内部状态 ----
        self._current_joints[arm] = current_joints

        # ---- 组装结果 ----
        total_duration = len(all_trajectory) * 0.02  # dt = 0.02s
        elapsed = time.time() - t0

        return PlanResult(
            success=True,
            arm=arm,
            trajectory=all_trajectory,
            duration=round(total_duration, 4),
            n_waypoints=len(all_trajectory),
            target_joints=current_joints,
            gripper_command=0.0,  # 序列结束时夹爪闭合
            phase="grasp_sequence",
            error_message="",
        )

    # ================================================================
    # 公开接口 4：get_current_state —— 获取当前关节状态
    # ================================================================

    def get_current_state(self) -> Dict:
        """
        获取当前双臂关节状态。

        返回：
            {
                "left":  [j1, j2, j3, j4, j5, j6],
                "right": [j1, j2, j3, j4, j5, j6],
            }
        """
        return {
            "left": list(self._current_joints["left"]),
            "right": list(self._current_joints["right"]),
        }

    # ================================================================
    # 公开接口 5：update_scene —— 更新场景
    # ================================================================

    def update_scene(self, obstacles: List[Dict]):
        """
        更新场景障碍物。

        参数：
            obstacles: 障碍物列表
                [{"name": str, "position": [x,y,z], "dimensions": [w,h,d]}]
        """
        self._planner.update_obstacles(obstacles)

    # ================================================================
    # 公开接口 6：set_joint_state —— 设置当前关节状态
    # ================================================================

    def set_joint_state(self, arm: str, joints: List[float]):
        """
        手动设置当前关节状态（用于同步机器人实际状态）。

        参数：
            arm:    "left" 或 "right"
            joints: 6 个关节角度
        """
        if arm not in ("left", "right"):
            raise ValueError(f"arm 必须是 'left' 或 'right'，收到 '{arm}'")
        if len(joints) != 6:
            raise ValueError(f"需要 6 个关节角度，收到 {len(joints)}")
        self._current_joints[arm] = list(joints)

    # ================================================================
    # 公开接口 7：transform_camera_to_base —— 坐标转换
    # ================================================================

    def transform_camera_to_base(self, u: float, v: float, depth: float,
                                  camera: str,
                                  joint_angles: Optional[List[float]] = None) -> List[float]:
        """
        将相机像素坐标转换为基座坐标。

        参数：
            u, v:       像素坐标
            depth:      深度（米）
            camera:     相机名称 "top" / "left_wrist" / "right_wrist"
            joint_angles: 关节角度（手腕相机必须提供）

        返回：
            [x, y, z] 基座坐标
        """
        joints_np = np.array(joint_angles) if joint_angles else None
        p = self._transformer.transform(u, v, depth, camera, joints_np)
        return p.tolist()

    # ================================================================
    # 内部方法：参数校验
    # ================================================================

    def _validate(self, target_pose: Dict, arm: str) -> Optional[PlanResult]:
        """
        校验 plan() 的输入参数。
        校验通过返回 None，失败返回 PlanResult 错误对象。
        """
        # 校验 arm
        if arm not in ("left", "right"):
            return PlanResult(
                success=False, arm=arm,
                error_message=f"arm 必须是 'left' 或 'right'，收到 '{arm}'",
            )

        # 校验 target_pose 格式
        if not isinstance(target_pose, dict):
            return PlanResult(
                success=False, arm=arm,
                error_message=f"target_pose 必须是字典，收到 {type(target_pose).__name__}",
            )
        if "position" not in target_pose:
            return PlanResult(
                success=False, arm=arm,
                error_message="target_pose 缺少 'position' 字段",
            )
        if "quaternion" not in target_pose:
            return PlanResult(
                success=False, arm=arm,
                error_message="target_pose 缺少 'quaternion' 字段",
            )

        pos = target_pose["position"]
        quat = target_pose["quaternion"]

        if len(pos) != 3:
            return PlanResult(
                success=False, arm=arm,
                error_message=f"position 需要 3 个元素，收到 {len(pos)}",
            )
        if len(quat) != 4:
            return PlanResult(
                success=False, arm=arm,
                error_message=f"quaternion 需要 4 个元素，收到 {len(quat)}",
            )

        # 粗略工作空间检查：X-Trainer 臂展约 0.4m
        # 目标距离基座超过 0.5m 基本不可达
        dist = np.linalg.norm(pos)
        if dist > 0.5:
            return PlanResult(
                success=False, arm=arm,
                error_message=(
                    f"目标距离基座 {dist:.3f}m，超出工作空间（~0.4m）。"
                    f"position={[round(v,3) for v in pos]}"
                ),
            )

        # 校验通过
        return None


# ============================================================
# 第三部分：测试函数
# ============================================================

def test_planner_interface():
    """
    测试运动规划统一接口。

    测试内容：
      1. 模块初始化 + 预热
      2. plan() 单臂规划
      3. plan_grasp_sequence() 抓取序列
      4. JSON 序列化
      5. 坐标转换
      6. 异常处理
    """
    print("=" * 60)
    print("运动规划统一接口测试")
    print("=" * 60)

    all_pass = True

    # ---- 测试 1：初始化 + 预热 ----
    print("\n[测试 1] 初始化 + 预热")
    try:
        module = MotionPlanningModule()
        module.warmup()
        print("  结果: PASS")
    except Exception as e:
        print(f"  失败: {e}")
        return False

    # ---- 测试 2：plan() 左臂规划 ----
    print("\n[测试 2] plan() 左臂规划")
    target = {"position": [0.3, -0.15, 0.25], "quaternion": [1, 0, 0, 0]}
    obstacles = [
        {"name": "桌子", "position": [0.4, 0, 0.15], "dimensions": [0.8, 0.6, 0.03]},
    ]
    result = module.plan(target, obstacles=obstacles, arm="left")
    passed = result.success
    all_pass = all_pass and passed
    if passed:
        print(f"  轨迹点: {result.n_waypoints}")
        print(f"  时长: {result.duration:.2f}s")
        print(f"  目标关节: {[f'{a:.4f}' for a in result.target_joints]}")
    else:
        print(f"  错误: {result.error_message}")
    print(f"  结果: {'PASS' if passed else 'FAIL'}")

    # ---- 测试 3：plan() 右臂规划 ----
    print("\n[测试 3] plan() 右臂规划")
    target_r = {"position": [0.3, 0.15, 0.25], "quaternion": [1, 0, 0, 0]}
    result_r = module.plan(target_r, arm="right")
    passed = result_r.success
    all_pass = all_pass and passed
    print(f"  成功: {result_r.success}")
    print(f"  结果: {'PASS' if passed else 'FAIL'}")

    # ---- 测试 4：plan_grasp_sequence() ----
    print("\n[测试 4] plan_grasp_sequence() 完整抓取序列")
    grasp_pose = {"position": [0.35, -0.15, 0.18], "quaternion": [1, 0, 0, 0]}
    place_pose = {"position": [0.35, 0.15, 0.30], "quaternion": [1, 0, 0, 0]}
    result_seq = module.plan_grasp_sequence(grasp_pose, place_pose, obstacles, arm="left")
    passed = result_seq.success
    all_pass = all_pass and passed
    if passed:
        print(f"  总轨迹点: {result_seq.n_waypoints}")
        print(f"  总时长: {result_seq.duration:.2f}s")
    else:
        print(f"  错误: {result_seq.error_message}")
    print(f"  结果: {'PASS' if passed else 'FAIL'}")

    # ---- 测试 5：JSON 序列化 ----
    print("\n[测试 5] JSON 序列化")
    json_str = result.to_json()
    restored = PlanResult.from_dict(json.loads(json_str))
    passed = restored.success == result.success and restored.arm == result.arm
    all_pass = all_pass and passed
    print(f"  JSON 长度: {len(json_str)} 字符")
    print(f"  反序列化: arm={restored.arm}, success={restored.success}")
    print(f"  结果: {'PASS' if passed else 'FAIL'}")

    # ---- 测试 6：异常处理 ----
    print("\n[测试 6] 异常处理")
    # 超出工作空间
    bad_pose = {"position": [1.0, 0, 0], "quaternion": [1, 0, 0, 0]}
    result_bad = module.plan(bad_pose, arm="left")
    passed = not result_bad.success and "工作空间" in result_bad.error_message
    all_pass = all_pass and passed
    print(f"  超出工作空间: success={result_bad.success}, msg={result_bad.error_message[:40]}...")
    # 缺少字段
    result_bad2 = module.plan({"position": [0.3, 0, 0.3]}, arm="left")
    passed2 = not result_bad2.success
    print(f"  缺少字段: success={result_bad2.success}, msg={result_bad2.error_message[:40]}")
    all_pass = all_pass and passed2
    print(f"  结果: {'PASS' if passed and passed2 else 'FAIL'}")

    # ---- 测试 7：坐标转换 ----
    print("\n[测试 7] 坐标转换")
    p_base = module.transform_camera_to_base(320, 240, 0.5, "top")
    passed = len(p_base) == 3 and all(np.isfinite(p_base))
    all_pass = all_pass and passed
    print(f"  俯视相机 (320,240,0.5m) → {p_base}")
    print(f"  结果: {'PASS' if passed else 'FAIL'}")

    # ---- 总结 ----
    print("\n" + "=" * 60)
    if all_pass:
        print("全部测试通过！")
    else:
        print("部分测试失败")
    print("=" * 60)

    return all_pass


# ============================================================
# 主函数入口
# ============================================================

if __name__ == "__main__":
    test_planner_interface()
