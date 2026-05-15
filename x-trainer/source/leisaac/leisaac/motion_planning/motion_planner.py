"""
cuRobo 避障轨迹规划模块
========================

功能：
  使用 cuRobo GPU 加速轨迹规划器，为 X-Trainer 双臂机器人生成无碰撞轨迹。
  从起始关节角度到目标位姿，自动避障、避自碰撞，输出平滑的关节轨迹。

cuRobo 轨迹规划原理：
  cuRobo 将轨迹规划转化为轨迹优化问题：
  1. 将路径参数化为一系列 knot points（时间节点）
  2. 同时优化：平滑性（速度/加速度/加加速度代价）+ 可行性（碰撞避障、关节限位）
  3. 多个轨迹种子在 GPU 上并行优化，选择最优的无碰撞结果

依赖：curobo, torch, numpy
"""

import torch
import numpy as np
from typing import Dict, List, Optional

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import JointState, Pose, GoalToolPose
from curobo.scene import Cuboid, Scene
from .curobo_config import build_curobo_robot_config


_DEFAULT_COLLISION_CACHE = {"cuboid": 32}


# ============================================================
# 第一部分：场景构建器
# ============================================================

class SceneBuilder:
    """
    场景构建器 —— 快速构建比赛场景中的障碍物。

    用法：
        >>> scene = SceneBuilder()
        >>> scene.add_table("桌子", [0.4, 0, 0.15], [0.8, 0.6, 0.03])
        >>> scene.add_box("盒子", [0.3, 0.2, 0.25], [0.05, 0.05, 0.05])
        >>> scene.add_basket("篮子", [0.5, -0.3, 0.2], [0.15, 0.15, 0.1])
        >>> obstacles = scene.get_scene()
    """

    def __init__(self):
        # 内部维护一个 Cuboid 列表
        self._cuboids: List[Cuboid] = []

    # ----------------------------------------------------------
    # add_table —— 添加桌子
    # ----------------------------------------------------------
    def add_table(self, name: str, position: list, dimensions: list):
        """
        添加桌子障碍物。

        参数：
            name:      桌子名称（唯一标识）
            position:  桌面中心位置 [x, y, z]，单位：米
            dimensions: 桌面尺寸 [长, 宽, 厚]，单位：米
        """
        self._add_cuboid(name, position, dimensions)

    # ----------------------------------------------------------
    # add_box —— 添加盒子
    # ----------------------------------------------------------
    def add_box(self, name: str, position: list, dimensions: list):
        """
        添加盒子障碍物。

        参数：
            name:      盒子名称
            position:  盒子中心位置 [x, y, z]
            dimensions: 盒子尺寸 [长, 宽, 高]
        """
        self._add_cuboid(name, position, dimensions)

    # ----------------------------------------------------------
    # add_basket —— 添加篮子
    # ----------------------------------------------------------
    def add_basket(self, name: str, position: list, dimensions: list):
        """
        添加篮子障碍物（简化为立方体外壳）。

        参数：
            name:      篮子名称
            position:  篮子中心位置 [x, y, z]
            dimensions: 篮子尺寸 [长, 宽, 高]
        """
        self._add_cuboid(name, position, dimensions)

    # ----------------------------------------------------------
    # 内部方法：添加立方体
    # ----------------------------------------------------------
    def _add_cuboid(self, name: str, position: list, dimensions: list,
                    quaternion: list = None):
        """
        内部方法：构造 Cuboid 对象并加入列表。

        cuRobo Cuboid 的 pose 格式: [x, y, z, qw, qx, qy, qz]
        默认无旋转: qw=1, qx=qy=qz=0
        """
        if quaternion is None:
            quaternion = [1.0, 0.0, 0.0, 0.0]
        pose = position + quaternion  # 拼接成 7 元素列表
        cuboid = Cuboid(name=name, pose=pose, dims=dimensions)
        self._cuboids.append(cuboid)

    # ----------------------------------------------------------
    # get_scene —— 获取 cuRobo Scene 对象
    # ----------------------------------------------------------
    def get_scene(self) -> Scene:
        """返回包含所有障碍物的 Scene 对象。"""
        return Scene(cuboid=list(self._cuboids))

    # ----------------------------------------------------------
    # clear —— 清除所有障碍物
    # ----------------------------------------------------------
    def clear(self):
        """清除所有障碍物。"""
        self._cuboids.clear()

    def __len__(self):
        return len(self._cuboids)

    def __repr__(self):
        return f"SceneBuilder(obstacles={len(self._cuboids)})"


# ============================================================
# 第二部分：双臂运动规划器
# ============================================================

class DualArmMotionPlanner:
    """
    双臂运动规划器。

    封装 cuRobo 的 MotionPlanner，提供双臂避障轨迹规划接口。

    使用流程：
        >>> planner = DualArmMotionPlanner()
        >>> planner.warmup()
        >>> result = planner.plan_to_pose("left", target_pose, current_joints)
        >>> if result["success"]:
        ...     print(f"轨迹 {result['n_waypoints']} 个点，时长 {result['duration']:.2f}s")
    """

    def __init__(self, robot_config: str = "xtrainer.yml",
                 scene_model: Optional[str] = None,
                 device: str = None):
        """
        初始化运动规划器。

        参数：
            robot_config: 机器人配置文件名或绝对路径
            scene_model:  场景配置文件名（可选，不传则无场景障碍物）
            device:       计算设备，默认自动选择 ("cuda" 如果有 GPU，否则 "cpu")。
        """
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        robot_config = build_curobo_robot_config(robot_config)
        # ---- 创建规划器配置 ----
        if scene_model:
            config = MotionPlannerCfg.create(
                robot=robot_config,
                scene_model=scene_model,
                collision_cache=_DEFAULT_COLLISION_CACHE,
            )
        else:
            config = MotionPlannerCfg.create(
                robot=robot_config,
                collision_cache=_DEFAULT_COLLISION_CACHE,
            )

        # ---- 创建 MotionPlanner 实例 ----
        self._planner = MotionPlanner(config)

        # ---- 记录关节和工具帧信息 ----
        self._joint_names = list(self._planner.joint_names)
        self._tool_frames = list(self._planner.tool_frames)

        # 识别左右臂 tool frame
        self._left_frame = None
        self._right_frame = None
        for frame in self._tool_frames:
            if "left" in frame:
                self._left_frame = frame
            elif "right" in frame:
                self._right_frame = frame

        # 场景构建器
        self._scene_builder = SceneBuilder()

        # 插值时间步长（从 cuRobo 配置中读取）
        self._interp_dt = self._planner.trajopt_solver.config.interpolation_dt
        self._home_tool_poses = self._compute_home_tool_poses()

        print(f"[运动规划器] 初始化完成")
        print(f"  关节: {self._joint_names}")
        print(f"  工具帧: {self._tool_frames}")
        print(f"  插值 dt: {self._interp_dt:.4f}s")

    # ================================================================
    # 公开接口 1：warmup —— 预热（编译 CUDA 图）
    # ================================================================

    def warmup(self, num_iterations: int = 5):
        """
        预热规划器。

        cuRobo 会预编译 CUDA 图以加速后续规划调用。
        首次规划前调用一次即可。

        参数：
            num_iterations: 预热迭代次数，推荐 5
        """
        print("[运动规划器] 预热中（首次编译 CUDA 图，约 30 秒）...")
        self._planner.warmup(enable_graph=True, num_warmup_iterations=num_iterations)
        print("[运动规划器] 预热完成")

    # ================================================================
    # 公开接口 2：plan_to_pose —— 单臂规划到目标位姿
    # ================================================================

    def plan_to_pose(self, arm: str, target_pose: Dict,
                     current_joints: list,
                     obstacles: Optional[List[Dict]] = None) -> Dict:
        """
        单臂规划：从当前关节角度到目标末端位姿，生成无碰撞轨迹。

        参数：
            arm:            "left" 或 "right"
            target_pose:    目标位姿 {"position": [x,y,z], "quaternion": [qw,qx,qy,qz]}
            current_joints: 当前 6 个关节角度（弧度）
            obstacles:      可选的临时障碍物列表，格式：
                            [{"name": str, "position": [x,y,z], "dimensions": [w,h,d]}]

        返回：
            {
                "success": bool,
                "trajectory": [[j1,j2,...,j6], ...] 或 None,  # 插值后的轨迹点
                "duration": float,     # 轨迹时长（秒）
                "n_waypoints": int,    # 轨迹点数量
                "error_message": str,  # 失败原因
            }
        """
        # 确定目标 tool frame
        target_frame = self._get_target_frame(arm)

        # 如果有临时障碍物，先更新场景
        if obstacles:
            for obs in obstacles:
                self._scene_builder.add_box(
                    obs["name"], obs["position"], obs["dimensions"]
                )
            self._planner.update_world(self._scene_builder.get_scene())

        # 构造起始关节状态
        q_start = self._make_joint_state(current_joints, arm)

        # 构造目标位姿（cuRobo 需要 5D 张量）
        goal = self._make_goal_pose(target_frame, target_pose)

        # ---- 调用 cuRobo 规划 ----
        result = self._planner.plan_pose(goal, q_start)

        # ---- 解析结果 ----
        if result is not None and result.success.any():
            interpolated = result.get_interpolated_plan()
            # 提取目标臂的 6 个关节
            joint_indices = self._get_joint_indices(arm)
            traj_full = interpolated.position.squeeze(0).cpu().numpy()  # (N, 12)
            traj_arm = traj_full[:, joint_indices]  # (N, 6)
            n_wp = traj_arm.shape[0]

            return {
                "success": True,
                "trajectory": traj_arm.tolist(),
                "duration": round(n_wp * self._interp_dt, 4),
                "n_waypoints": n_wp,
                "error_message": "",
            }
        else:
            p = target_pose["position"]
            return {
                "success": False,
                "trajectory": None,
                "duration": 0.0,
                "n_waypoints": 0,
                "error_message": (
                    f"{arm}臂规划失败：目标 [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}] "
                    f"可能超出工作空间或与障碍物冲突"
                ),
            }

    # ================================================================
    # 公开接口 3：plan_joint_to_joint —— 关节空间到关节空间规划
    # ================================================================

    def plan_joint_to_joint(self, arm: str,
                            start_joints: list,
                            goal_joints: list) -> Dict:
        """
        关节空间到关节空间规划。

        与 plan_to_pose 不同，这里直接指定起始和目标关节角度，
        不经过 IK 求解。适用于已知安全关节角的场景（如回到 home 位）。

        参数：
            arm:          "left" 或 "right"
            start_joints: 起始 6 个关节角度
            goal_joints:  目标 6 个关节角度

        返回：
            同 plan_to_pose
        """
        # 构造起始状态
        q_start = self._make_joint_state(start_joints, arm)

        # 构造目标状态（作为 goalset）
        full_goal = np.zeros(12, dtype=np.float32)
        indices = self._get_joint_indices(arm)
        full_goal[indices] = goal_joints

        # cuRobo plan_pose 也支持 JointState 作为目标
        # 但更简单的方式是用 FK 得到目标位姿，再用 plan_to_pose
        # 这里我们直接用 goal_joint_state 方式
        goal_js = JointState.from_position(
            torch.tensor([full_goal], dtype=torch.float32, device=self._device),
            joint_names=self._joint_names,
        )

        # ---- 调用 cuRobo 规划 ----
        # plan_joint 需要 goal 也是 JointState
        result = self._planner.plan_single(goal_js, q_start)

        if result is not None and result.success.any():
            interpolated = result.get_interpolated_plan()
            traj_full = interpolated.position.squeeze(0).cpu().numpy()
            traj_arm = traj_full[:, indices]
            n_wp = traj_arm.shape[0]

            return {
                "success": True,
                "trajectory": traj_arm.tolist(),
                "duration": round(n_wp * self._interp_dt, 4),
                "n_waypoints": n_wp,
                "error_message": "",
            }
        else:
            return {
                "success": False,
                "trajectory": None,
                "duration": 0.0,
                "n_waypoints": 0,
                "error_message": f"{arm}臂关节空间规划失败，路径可能与障碍物冲突",
            }

    # ================================================================
    # 公开接口 4：plan_grasp —— 三阶段抓取规划
    # ================================================================

    def plan_grasp(self, arm: str, grasp_pose: Dict,
                   current_joints: list,
                   approach_offset: float = 0.1,
                   lift_offset: float = 0.1) -> Dict:
        """
        三阶段抓取规划：接近 → 抓取 → 提升。

        cuRobo 的 plan_grasp 自动规划三段轨迹：
          1. Approach: 从当前位置到预抓取点（沿接近轴偏移 approach_offset）
          2. Grasp:    从预抓取点到最终抓取点
          3. Lift:     从抓取点沿提升轴提升 lift_offset

        参数：
            arm:              "left" 或 "right"
            grasp_pose:       抓取位姿 {"position": [x,y,z], "quaternion": [qw,qx,qy,qz]}
            current_joints:   当前 6 个关节角度
            approach_offset:  接近偏移距离（米），默认 0.1
            lift_offset:      提升偏移距离（米），默认 0.1

        返回：
            {
                "success": bool,
                "approach_trajectory": [[j1,...,j6], ...] 或 None,
                "grasp_trajectory":    [[j1,...,j6], ...] 或 None,
                "lift_trajectory":     [[j1,...,j6], ...] 或 None,
                "total_duration": float,
                "total_waypoints": int,
                "error_message": str,
            }
        """
        target_frame = self._get_target_frame(arm)
        joint_indices = self._get_joint_indices(arm)

        # 构造起始状态
        q_start = self._make_joint_state(current_joints, arm)

        # 构造抓取位姿（GoalToolPose 需要 5D 张量）
        grasp_goal = self._make_goal_pose(target_frame, grasp_pose)

        # ---- 调用 cuRobo 三阶段抓取规划 ----
        result = self._planner.plan_grasp(
            current_state=q_start,
            grasp_poses=grasp_goal,
            grasp_approach_offset=approach_offset,
            grasp_lift_offset=lift_offset,
            plan_approach_to_grasp=True,
            plan_grasp_to_lift=True,
            grasp_lift_in_tool_frame=True,
        )

        # ---- 解析结果 ----
        if result.success is not None and result.success.any():
            approach_traj = self._extract_trajectory(
                result.approach_interpolated_trajectory, joint_indices
            )
            grasp_traj = self._extract_trajectory(
                result.grasp_interpolated_trajectory, joint_indices
            )
            lift_traj = self._extract_trajectory(
                result.lift_interpolated_trajectory, joint_indices
            )

            total_wp = sum(len(t) for t in [approach_traj, grasp_traj, lift_traj] if t)

            return {
                "success": True,
                "approach_trajectory": approach_traj,
                "grasp_trajectory": grasp_traj,
                "lift_trajectory": lift_traj,
                "total_duration": round(total_wp * self._interp_dt, 4),
                "total_waypoints": total_wp,
                "error_message": "",
            }
        else:
            p = grasp_pose["position"]
            return {
                "success": False,
                "approach_trajectory": None,
                "grasp_trajectory": None,
                "lift_trajectory": None,
                "total_duration": 0.0,
                "total_waypoints": 0,
                "error_message": (
                    f"{arm}臂抓取规划失败：目标 [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}] "
                    f"可能不可达或与障碍物冲突"
                ),
            }

    # ================================================================
    # 公开接口 5：update_obstacles —— 运行时更新障碍物
    # ================================================================

    def update_obstacles(self, obstacles: List[Dict]):
        """
        运行时更新场景障碍物（不重建规划器）。

        参数：
            obstacles: 障碍物列表，每个元素：
                {
                    "name": str,
                    "position": [x, y, z],
                    "dimensions": [w, h, d],
                }
        """
        # 先清空旧障碍物
        self._scene_builder.clear()
        # 添加新障碍物
        for obs in obstacles:
            self._scene_builder.add_box(
                obs["name"], obs["position"], obs["dimensions"]
            )
        # 更新到规划器
        self._planner.update_world(self._scene_builder.get_scene())

    # ================================================================
    # 公开接口 6：get_scene_builder —— 获取场景构建器
    # ================================================================

    def get_scene_builder(self) -> SceneBuilder:
        """获取场景构建器，用于逐步构建场景。"""
        return self._scene_builder

    def apply_scene(self):
        """将场景构建器中的障碍物应用到规划器。"""
        self._planner.update_world(self._scene_builder.get_scene())

    # ================================================================
    # 内部方法
    # ================================================================

    def _get_target_frame(self, arm: str) -> str:
        """根据手臂名获取对应的 tool frame 名称。"""
        if arm == "left":
            if self._left_frame is None:
                raise ValueError("未找到左臂 tool frame，请检查 xtrainer.yml 配置")
            return self._left_frame
        elif arm == "right":
            if self._right_frame is None:
                raise ValueError("未找到右臂 tool frame，请检查 xtrainer.yml 配置")
            return self._right_frame
        else:
            raise ValueError(f"arm 必须是 'left' 或 'right'，收到 '{arm}'")

    def _get_joint_indices(self, arm: str) -> list:
        """
        获取目标臂在 12 关节 cspace 中的索引。
        cspace 顺序: [J1_1~J1_6, J2_1~J2_6]
        左臂索引 0~5，右臂索引 6~11
        """
        if arm == "left":
            return list(range(6))
        else:
            return list(range(6, 12))

    def _make_joint_state(self, joints: list, arm: str = "left") -> JointState:
        """
        构造 cuRobo JointState。
        将 6 关节角度扩展为 12 关节（另一半补零）。
        """
        full = np.zeros(12, dtype=np.float32)
        full[self._get_joint_indices(arm)] = joints
        return JointState.from_position(
            torch.as_tensor(full[None, :], dtype=torch.float32, device=self._device),
            joint_names=self._joint_names,
        )

    def forward_kinematics(self, joint_angles: list, arm: str = "left") -> Dict:
        """
        正运动学：从目标臂 6 个关节角计算末端位姿。

        主要用于测试和调试：先用 FK 生成当前 URDF 确认可达的目标，
        再把这个目标交给 IK/规划器验证，避免测试坐标与机器人模型脱节。
        """
        full = np.zeros(12, dtype=np.float32)
        full[self._get_joint_indices(arm)] = joint_angles
        joint_state = JointState.from_position(
            torch.as_tensor(full[None, :], dtype=torch.float32, device=self._device),
            joint_names=self._joint_names,
        )
        target_frame = self._get_target_frame(arm)
        kin = self._planner.compute_kinematics(joint_state)
        tool_pose = kin.tool_poses.get_link_pose(target_frame)
        return {
            "position": tool_pose.position.squeeze().detach().cpu().numpy().tolist(),
            "quaternion": tool_pose.quaternion.squeeze().detach().cpu().numpy().tolist(),
        }

    def _make_goal_pose(self, target_frame: str, target_pose: Dict) -> GoalToolPose:
        """
        构造 cuRobo GoalToolPose。
        target_pose: {"position": [x,y,z], "quaternion": [qw,qx,qy,qz]}
        """
        pos = torch.tensor(
            [target_pose["position"]], dtype=torch.float32, device=self._device
        )
        quat = torch.tensor(
            [target_pose["quaternion"]], dtype=torch.float32, device=self._device
        )
        goal_dict = self._make_single_arm_goal_dict(target_frame, pos, quat)
        return GoalToolPose.from_poses(
            goal_dict, ordered_tool_frames=self._tool_frames, num_goalset=1
        )

    def _compute_home_tool_poses(self) -> Dict[str, Pose]:
        home_state = JointState.from_position(
            torch.zeros(
                (1, len(self._joint_names)), dtype=torch.float32, device=self._device
            ),
            joint_names=self._joint_names,
        )
        return self._planner.compute_kinematics(home_state).tool_poses.to_dict()

    def _make_single_arm_goal_dict(
        self, target_frame: str, position: torch.Tensor, quaternion: torch.Tensor
    ) -> Dict[str, Pose]:
        goal_dict = {
            frame: Pose(
                position=pose.position.clone(),
                quaternion=pose.quaternion.clone(),
                name=frame,
                normalize_rotation=False,
            )
            for frame, pose in self._home_tool_poses.items()
        }
        goal_dict[target_frame] = Pose(
            position=position,
            quaternion=quaternion,
            name=target_frame,
            normalize_rotation=False,
        )
        return goal_dict

    def _extract_trajectory(self, curobo_traj, joint_indices: list) -> Optional[list]:
        """
        从 cuRobo 轨迹对象中提取目标臂的关节轨迹。
        返回 list of list[float]，如果轨迹为 None 则返回 None。
        """
        if curobo_traj is None:
            return None
        pos = curobo_traj.position.reshape(-1, curobo_traj.position.shape[-1])
        traj_np = pos.cpu().numpy()  # (N, 12)
        traj_arm = traj_np[:, joint_indices]  # (N, 6)
        return traj_arm.tolist()


# ============================================================
# 第三部分：测试函数
# ============================================================

def test_motion_planner():
    """
    测试运动规划器全部功能。

    测试内容：
      1. 初始化 + 场景构建
      2. 预热
      3. plan_to_pose（左臂 / 右臂）
      4. plan_joint_to_joint
      5. plan_grasp（三阶段）
      6. update_obstacles（运行时更新）
    """
    print("=" * 60)
    print("cuRobo 运动规划器测试")
    print("=" * 60)

    all_pass = True

    # ---- 测试 1：初始化 + 场景构建 ----
    print("\n[测试 1] 初始化 + 场景构建")
    try:
        planner = DualArmMotionPlanner(robot_config="xtrainer.yml")
        # 构建比赛场景
        builder = planner.get_scene_builder()
        builder.add_table("桌子", [0.4, 0, 0.15], [0.8, 0.6, 0.03])
        builder.add_box("障碍盒", [0.3, 0.1, 0.25], [0.05, 0.05, 0.1])
        builder.add_basket("篮子", [0.5, -0.3, 0.2], [0.15, 0.15, 0.1])
        planner.apply_scene()
        print(f"  障碍物数量: {len(builder)}")
        print("  结果: PASS")
    except Exception as e:
        print(f"  失败: {e}")
        print("  结果: FAIL")
        return False

    # ---- 测试 2：预热 ----
    print("\n[测试 2] 规划器预热")
    try:
        planner.warmup()
        print("  结果: PASS")
    except Exception as e:
        print(f"  预热失败: {e}")
        print("  结果: FAIL")
        all_pass = False

    # ---- 测试 3：左臂 plan_to_pose ----
    print("\n[测试 3] 左臂 plan_to_pose")
    left_target = {
        "position": [0.3, -0.15, 0.25],
        "quaternion": [1.0, 0.0, 0.0, 0.0],
    }
    result_l = planner.plan_to_pose("left", left_target, current_joints=[0]*6)
    passed = result_l["success"]
    all_pass = all_pass and passed
    if passed:
        print(f"  轨迹点数: {result_l['n_waypoints']}")
        print(f"  轨迹时长: {result_l['duration']:.2f}s")
        print(f"  首个轨迹点: {[f'{a:.4f}' for a in result_l['trajectory'][0]]}")
    else:
        print(f"  错误: {result_l['error_message']}")
    print(f"  结果: {'PASS' if passed else 'FAIL'}")

    # ---- 测试 4：右臂 plan_to_pose ----
    print("\n[测试 4] 右臂 plan_to_pose")
    right_target = {
        "position": [0.3, 0.15, 0.25],
        "quaternion": [1.0, 0.0, 0.0, 0.0],
    }
    result_r = planner.plan_to_pose("right", right_target, current_joints=[0]*6)
    passed = result_r["success"]
    all_pass = all_pass and passed
    if passed:
        print(f"  轨迹点数: {result_r['n_waypoints']}")
        print(f"  轨迹时长: {result_r['duration']:.2f}s")
    else:
        print(f"  错误: {result_r['error_message']}")
    print(f"  结果: {'PASS' if passed else 'FAIL'}")

    # ---- 测试 5：plan_grasp（三阶段） ----
    print("\n[测试 5] 左臂 plan_grasp（接近→抓取→提升）")
    grasp_pose = {
        "position": [0.35, -0.15, 0.18],
        "quaternion": [1.0, 0.0, 0.0, 0.0],
    }
    result_g = planner.plan_grasp(
        "left", grasp_pose, current_joints=[0]*6,
        approach_offset=0.1, lift_offset=0.1,
    )
    passed = result_g["success"]
    all_pass = all_pass and passed
    if passed:
        print(f"  接近段: {len(result_g['approach_trajectory'])} 点")
        print(f"  抓取段: {len(result_g['grasp_trajectory'])} 点")
        print(f"  提升段: {len(result_g['lift_trajectory'])} 点")
        print(f"  总时长: {result_g['total_duration']:.2f}s")
    else:
        print(f"  错误: {result_g['error_message']}")
    print(f"  结果: {'PASS' if passed else 'FAIL'}")

    # ---- 测试 6：update_obstacles ----
    print("\n[测试 6] 运行时更新障碍物")
    new_obstacles = [
        {"name": "新桌子", "position": [0.4, 0, 0.15], "dimensions": [0.6, 0.4, 0.03]},
        {"name": "新盒子", "position": [0.2, 0.2, 0.2], "dimensions": [0.04, 0.04, 0.04]},
    ]
    planner.update_obstacles(new_obstacles)
    # 再次规划，验证更新后的场景
    result_u = planner.plan_to_pose("left", left_target, current_joints=[0]*6)
    passed = result_u["success"]
    all_pass = all_pass and passed
    if passed:
        print(f"  更新后规划成功: {result_u['n_waypoints']} 点")
    else:
        print(f"  更新后规划失败: {result_u['error_message']}")
    print(f"  结果: {'PASS' if passed else 'FAIL'}")

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
    test_motion_planner()
