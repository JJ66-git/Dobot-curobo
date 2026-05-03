"""
cuRobo 逆运动学（IK）求解模块
==============================

功能：
  使用 cuRobo GPU 加速的 IK 求解器，为 X-Trainer 双臂机器人求解逆运动学。
  输入末端执行器目标位姿（位置 + 四元数），输出 6 个关节角度。

cuRobo IK 原理：
  将 IK 转化为非线性优化问题，在 GPU 上并行运行多个随机种子（seeds），
  每个种子从不同初始关节角度出发，寻找使末端到达目标位姿的关节角度。
  最终选择误差最小的解。默认 32 个种子，成功率 > 95%。

依赖：curobo, torch, numpy
"""

import torch
import numpy as np
from typing import Dict, Optional

from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.types import Pose, GoalToolPose


# ============================================================
# 第一部分：关节限位常量
# ============================================================

# X-Trainer 每个臂 6 个关节，限位均为 ±π（来自 URDF）
JOINT_LIMITS_LOWER = np.full(6, -np.pi, dtype=np.float32)  # 下限 -3.14159
JOINT_LIMITS_UPPER = np.full(6,  np.pi, dtype=np.float32)  # 上限  3.14159


# ============================================================
# 第二部分：DualArmIKSolver 类
# ============================================================

class DualArmIKSolver:
    """
    双臂 IK 求解器。

    封装 cuRobo 的 InverseKinematics，提供简洁的双臂 IK 接口。

    使用流程：
        >>> solver = DualArmIKSolver()
        >>> target = {"position": [0.3, -0.2, 0.3], "quaternion": [1, 0, 0, 0]}
        >>> result = solver.solve_left_arm(target)
        >>> if result["success"]:
        ...     print(f"关节角度: {result['joint_angles']}")
    """

    def __init__(self, robot_config: str = "xtrainer.yml",
                 num_seeds: int = 32,
                 self_collision_check: bool = True):
        """
        初始化 IK 求解器。

        参数：
            robot_config: cuRobo 机器人配置文件。
                          可以是文件名（在 curobo/content/configs/robot/ 下搜索），
                          或 Step 2 生成的 xtrainer.yml 的绝对路径。
            num_seeds: 并行优化种子数。越多越准确但越慢，推荐 32。
            self_collision_check: 是否启用自碰撞检测，推荐 True。
        """
        self.num_seeds = num_seeds

        # ---- 创建 IK 配置 ----
        # InverseKinematicsCfg.create() 会解析 YAML 中的 URDF、碰撞球、关节空间等
        config = InverseKinematicsCfg.create(
            robot=robot_config,
            num_seeds=num_seeds,
            self_collision_check=self_collision_check,
        )

        # ---- 创建 IK 求解器 ----
        self._ik = InverseKinematics(config)

        # ---- 解析工具帧名称 ----
        # xtrainer.yml 配置了 tool_frames: [left_ee_link, right_ee_link]
        self._tool_frames = self._ik.tool_frames
        self._left_frame = None
        self._right_frame = None
        for frame in self._tool_frames:
            if "left" in frame:
                self._left_frame = frame
            elif "right" in frame:
                self._right_frame = frame

        # ---- 获取 cspace 中的关节名称 ----
        # xtrainer.yml 的 cspace.joint_names 为 [J1_1~J1_6, J2_1~J2_6]
        self._joint_names = list(self._ik.joint_names)

        print(f"[IK 求解器] 初始化完成")
        print(f"  工具帧: {self._tool_frames}")
        print(f"  左臂帧: {self._left_frame}")
        print(f"  右臂帧: {self._right_frame}")
        print(f"  关节名: {self._joint_names}")
        print(f"  种子数: {num_seeds}")
        print(f"  自碰撞检测: {self_collision_check}")

    # ================================================================
    # 公开接口 1：solve_left_arm —— 左臂 IK 求解
    # ================================================================

    def solve_left_arm(self, target_pose: Dict) -> Dict:
        """
        左臂 IK 求解。

        参数：
            target_pose: 目标位姿字典
                {
                    "position": [x, y, z],         # 基座坐标系，单位：米
                    "quaternion": [qw, qx, qy, qz] # cuRobo wxyz 约定
                }

        返回：
            {
                "success": True/False,
                "joint_angles": [j1, j2, j3, j4, j5, j6] 或 None,  # 弧度
                "position_error_mm": float,   # 位置误差，单位：毫米
                "error_message": str          # 失败时的友好提示
            }
        """
        return self._solve_single(target_pose, self._left_frame, "左臂")

    # ================================================================
    # 公开接口 2：solve_right_arm —— 右臂 IK 求解
    # ================================================================

    def solve_right_arm(self, target_pose: Dict) -> Dict:
        """
        右臂 IK 求解。

        参数：同 solve_left_arm
        返回：同 solve_left_arm
        """
        return self._solve_single(target_pose, self._right_frame, "右臂")

    # ================================================================
    # 公开接口 3：solve_both —— 双臂同时求解
    # ================================================================

    def solve_both(self, left_pose: Dict, right_pose: Dict) -> Dict:
        """
        双臂同时 IK 求解（单次 GPU 调用，效率更高）。

        参数：
            left_pose:  左臂目标位姿 {"position": [...], "quaternion": [...]}
            right_pose: 右臂目标位姿 {"position": [...], "quaternion": [...]}

        返回：
            {
                "left":  {"success": bool, "joint_angles": list, "position_error_mm": float, "error_message": str},
                "right": {"success": bool, "joint_angles": list, "position_error_mm": float, "error_message": str},
                "both_success": bool,
            }
        """
        # ---- 参数校验 ----
        self._validate_pose(left_pose, "左臂")
        self._validate_pose(right_pose, "右臂")

        # ---- 构造 cuRobo Pose 张量 ----
        # shape = (1, 3) 和 (1, 4)，batch 维度为 1
        left_pos = torch.tensor(
            [left_pose["position"]], dtype=torch.float32, device="cuda"
        )
        left_quat = torch.tensor(
            [left_pose["quaternion"]], dtype=torch.float32, device="cuda"
        )
        right_pos = torch.tensor(
            [right_pose["position"]], dtype=torch.float32, device="cuda"
        )
        right_quat = torch.tensor(
            [right_pose["quaternion"]], dtype=torch.float32, device="cuda"
        )

        # ---- 构造 GoalToolPose ----
        # 字典 key 是 tool_frame 名，value 是 Pose 对象
        goal_dict = {
            self._left_frame:  Pose(position=left_pos,  quaternion=left_quat),
            self._right_frame: Pose(position=right_pos, quaternion=right_quat),
        }
        goal = GoalToolPose.from_poses(goal_dict, num_goalset=1)

        # ---- GPU 求解 ----
        result = self._ik.solve_pose(goal)

        # ---- 解析结果 ----
        success = result.success.item()
        pos_error = result.position_error.item()          # 单位：米
        joint_full = result.js_solution.squeeze().cpu().numpy()  # (12,)

        # 拆分左右臂各 6 个关节
        left_joints = joint_full[:6].tolist()
        right_joints = joint_full[6:12].tolist()

        # 构造返回
        left_out = {
            "success": success,
            "joint_angles": left_joints if success else None,
            "position_error_mm": round(pos_error * 1000, 3),
            "error_message": "" if success else "双臂 IK 求解失败，目标可能超出工作空间",
        }
        right_out = {
            "success": success,
            "joint_angles": right_joints if success else None,
            "position_error_mm": round(pos_error * 1000, 3),
            "error_message": "" if success else "双臂 IK 求解失败，目标可能超出工作空间",
        }

        return {
            "left": left_out,
            "right": right_out,
            "both_success": success,
        }

    # ================================================================
    # 公开接口 4：check_ik_valid —— 验证关节角度是否在限位内
    # ================================================================

    def check_ik_valid(self, joint_angles: list, arm: str = "left") -> Dict:
        """
        验证关节角度是否在限位内。

        参数：
            joint_angles: 6 个关节角度（弧度）
            arm: "left" 或 "right"

        返回：
            {
                "valid": bool,
                "violations": [{"joint": str, "value": float, "limit": str}, ...]
            }
        """
        if len(joint_angles) != 6:
            return {
                "valid": False,
                "violations": [{"joint": "N/A", "value": 0, "limit": f"需要 6 个关节，收到 {len(joint_angles)} 个"}],
            }

        # 根据手臂确定关节名称前缀
        prefix = "J1" if arm == "left" else "J2"
        violations = []

        for i, (angle, lo, hi) in enumerate(
            zip(joint_angles, JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER)
        ):
            if angle < lo:
                violations.append({
                    "joint": f"{prefix}_{i+1}",
                    "value": round(float(angle), 4),
                    "limit": f"下限 {lo:.4f}",
                })
            elif angle > hi:
                violations.append({
                    "joint": f"{prefix}_{i+1}",
                    "value": round(float(angle), 4),
                    "limit": f"上限 {hi:.4f}",
                })

        return {
            "valid": len(violations) == 0,
            "violations": violations,
        }

    # ================================================================
    # 公开接口 5：forward_kinematics —— 正运动学验证
    # ================================================================

    def forward_kinematics(self, joint_angles: list, arm: str = "left") -> Dict:
        """
        正运动学：从关节角度计算末端执行器位姿。

        原理：cuRobo 内部维护了完整的运动学链，给定 12 个关节角度，
        可以计算出每个 tool_frame 在 base_link 下的位置和朝向。

        参数：
            joint_angles: 6 个关节角度（弧度）
            arm: "left" 或 "right"

        返回：
            {
                "position": [x, y, z],
                "quaternion": [qw, qx, qy, qz],
            }
        """
        # 构造完整的 12 关节状态张量
        # cspace 顺序: [J1_1~J1_6, J2_1~J2_6]
        full = np.zeros(12, dtype=np.float32)
        if arm == "left":
            full[:6] = joint_angles
            target_frame = self._left_frame
        else:
            full[6:12] = joint_angles
            target_frame = self._right_frame

        joint_state = torch.tensor(full, device="cuda", dtype=torch.float32)

        # cuRobo 正运动学
        kin = self._ik.compute_kinematics(joint_state)
        tool_pose = kin.tool_poses[target_frame]

        pos = tool_pose.position.squeeze().cpu().numpy().tolist()
        quat = tool_pose.quaternion.squeeze().cpu().numpy().tolist()

        return {
            "position": pos,
            "quaternion": quat,
        }

    # ================================================================
    # 内部方法：单臂求解
    # ================================================================

    def _solve_single(self, target_pose: Dict, target_frame: str,
                      arm_label: str) -> Dict:
        """
        单臂 IK 求解的内部实现。

        参数：
            target_pose: 目标位姿字典
            target_frame: cuRobo tool_frame 名（left_ee_link 或 right_ee_link）
            arm_label: 用于错误信息的标签（"左臂" 或 "右臂"）

        返回：
            标准结果字典
        """
        # ---- 参数校验 ----
        self._validate_pose(target_pose, arm_label)

        # ---- 构造 cuRobo Pose 张量 ----
        pos_tensor = torch.tensor(
            [target_pose["position"]], dtype=torch.float32, device="cuda"
        )
        quat_tensor = torch.tensor(
            [target_pose["quaternion"]], dtype=torch.float32, device="cuda"
        )

        # ---- 构造 GoalToolPose ----
        goal_dict = {target_frame: Pose(position=pos_tensor, quaternion=quat_tensor)}
        goal = GoalToolPose.from_poses(goal_dict, num_goalset=1)

        # ---- GPU 求解 ----
        result = self._ik.solve_pose(goal)

        # ---- 解析结果 ----
        success = result.success.item()
        pos_error_m = result.position_error.item()                     # 米
        pos_error_mm = round(pos_error_m * 1000, 3)                    # 毫米
        joint_full = result.js_solution.squeeze().cpu().numpy()        # (12,)

        # 提取目标臂的 6 个关节
        if arm_label == "左臂":
            joints = joint_full[:6].tolist()
        else:
            joints = joint_full[6:12].tolist()

        # ---- 友好错误信息 ----
        if success:
            error_msg = ""
        else:
            p = target_pose["position"]
            error_msg = (
                f"目标超出{arm_label}工作空间："
                f"position=[{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]，"
                f"位置误差 {pos_error_mm:.1f}mm。"
                f"请检查目标是否在机器人可达范围内。"
            )

        return {
            "success": success,
            "joint_angles": joints if success else None,
            "position_error_mm": pos_error_mm,
            "error_message": error_msg,
        }

    # ================================================================
    # 内部方法：参数校验
    # ================================================================

    @staticmethod
    def _validate_pose(pose: Dict, arm_label: str):
        """校验 target_pose 格式。"""
        if not isinstance(pose, dict):
            raise ValueError(f"{arm_label} target_pose 必须是字典，收到 {type(pose)}")
        if "position" not in pose:
            raise ValueError(f"{arm_label} target_pose 缺少 'position' 字段")
        if "quaternion" not in pose:
            raise ValueError(f"{arm_label} target_pose 缺少 'quaternion' 字段")
        if len(pose["position"]) != 3:
            raise ValueError(f"{arm_label} position 需要 3 个元素，收到 {len(pose['position'])}")
        if len(pose["quaternion"]) != 4:
            raise ValueError(f"{arm_label} quaternion 需要 4 个元素，收到 {len(pose['quaternion'])}")


# ============================================================
# 第三部分：测试函数
# ============================================================

def test_ik_solver():
    """
    测试 IK 求解器全部功能。

    测试内容：
      1. 求解器初始化
      2. 左臂 IK 求解
      3. 右臂 IK 求解
      4. 双臂同时求解
      5. 关节限位验证
      6. 正运动学验证（IK → FK 回到原位姿）
    """
    print("=" * 60)
    print("cuRobo IK 求解器测试")
    print("=" * 60)

    all_pass = True

    # ---- 测试 1：初始化 ----
    print("\n[测试 1] 求解器初始化")
    try:
        solver = DualArmIKSolver(
            robot_config="xtrainer.yml",
            num_seeds=32,
            self_collision_check=True,
        )
        print("  结果: PASS")
    except Exception as e:
        print(f"  初始化失败: {e}")
        print("  结果: FAIL")
        return False

    # ---- 测试 2：左臂 IK ----
    print("\n[测试 2] 左臂 IK 求解")
    left_target = {
        "position": [0.3, -0.2, 0.3],
        "quaternion": [1.0, 0.0, 0.0, 0.0],
    }
    result_l = solver.solve_left_arm(left_target)
    passed = result_l["success"]
    all_pass = all_pass and passed
    print(f"  目标: pos={left_target['position']}")
    print(f"  成功: {result_l['success']}")
    if passed:
        print(f"  关节角度: {[f'{a:.4f}' for a in result_l['joint_angles']]}")
        print(f"  位置误差: {result_l['position_error_mm']:.3f} mm")
    else:
        print(f"  错误: {result_l['error_message']}")
    print(f"  结果: {'PASS' if passed else 'FAIL'}")

    # ---- 测试 3：右臂 IK ----
    print("\n[测试 3] 右臂 IK 求解")
    right_target = {
        "position": [0.3, 0.2, 0.3],
        "quaternion": [1.0, 0.0, 0.0, 0.0],
    }
    result_r = solver.solve_right_arm(right_target)
    passed = result_r["success"]
    all_pass = all_pass and passed
    print(f"  目标: pos={right_target['position']}")
    print(f"  成功: {result_r['success']}")
    if passed:
        print(f"  关节角度: {[f'{a:.4f}' for a in result_r['joint_angles']]}")
        print(f"  位置误差: {result_r['position_error_mm']:.3f} mm")
    else:
        print(f"  错误: {result_r['error_message']}")
    print(f"  结果: {'PASS' if passed else 'FAIL'}")

    # ---- 测试 4：双臂同时求解 ----
    print("\n[测试 4] 双臂同时求解")
    result_both = solver.solve_both(left_target, right_target)
    passed = result_both["both_success"]
    all_pass = all_pass and passed
    print(f"  双臂成功: {result_both['both_success']}")
    if passed:
        print(f"  左臂关节: {[f'{a:.4f}' for a in result_both['left']['joint_angles']]}")
        print(f"  右臂关节: {[f'{a:.4f}' for a in result_both['right']['joint_angles']]}")
    print(f"  结果: {'PASS' if passed else 'FAIL'}")

    # ---- 测试 5：关节限位验证 ----
    print("\n[测试 5] 关节限位验证")
    # 合法关节角度
    valid_check = solver.check_ik_valid([0.1, -0.5, 0.3, 0.0, 0.0, 0.0], "left")
    print(f"  合法角度验证: valid={valid_check['valid']}")
    # 超限关节角度
    invalid_check = solver.check_ik_valid([0.1, -4.0, 0.3, 0.0, 0.0, 0.0], "left")
    print(f"  超限角度验证: valid={invalid_check['valid']}, violations={invalid_check['violations']}")
    passed = valid_check["valid"] and not invalid_check["valid"]
    all_pass = all_pass and passed
    print(f"  结果: {'PASS' if passed else 'FAIL'}")

    # ---- 测试 6：IK → FK 验证 ----
    print("\n[测试 6] IK → FK 一致性验证")
    if result_l["success"]:
        fk = solver.forward_kinematics(result_l["joint_angles"], "left")
        pos_error = np.linalg.norm(
            np.array(fk["position"]) - np.array(left_target["position"])
        )
        passed = pos_error < 0.01  # 误差 < 1cm
        print(f"  IK 目标: {left_target['position']}")
        print(f"  FK 结果: {[f'{v:.4f}' for v in fk['position']]}")
        print(f"  位置误差: {pos_error*1000:.3f} mm")
    else:
        passed = False
        print(f"  跳过（IK 未成功）")
    all_pass = all_pass and passed
    print(f"  结果: {'PASS' if passed else 'FAIL'}")

    # ---- 总结 ----
    print("\n" + "=" * 60)
    if all_pass:
        print("全部测试通过！")
    else:
        print("部分测试失败，请检查代码或目标位姿")
    print("=" * 60)

    return all_pass


# ============================================================
# 主函数入口
# ============================================================

if __name__ == "__main__":
    test_ik_solver()
