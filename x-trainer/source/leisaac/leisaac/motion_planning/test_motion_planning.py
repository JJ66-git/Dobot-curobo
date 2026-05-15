"""
运动规划模块端到端测试
======================

测试内容：
  1. test_coordinate_transform  —— 坐标转换精度
  2. test_ik_solver             —— IK 求解 + FK 验证
  3. test_motion_planner        —— 轨迹规划 + 平滑性检查
  4. test_state_machine         —— 状态机流转
  5. test_full_pipeline         —— 完整流水线

运行方式：
  cd x-trainer/source/leisaac
  python -m leisaac.motion_planning.test_motion_planning
"""

import sys
import time
import math
import numpy as np


# ============================================================
# 辅助函数
# ============================================================

def _header(title: str):
    """打印测试标题。"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def _sub(name: str):
    """打印子测试名称。"""
    print(f"\n  [{name}]")


def _ok(msg: str):
    """打印通过信息。"""
    print(f"    [OK] {msg}")


def _fail(msg: str):
    """打印失败信息。"""
    print(f"    [FAIL] {msg}")


def _info(msg: str):
    """打印信息。"""
    print(f"    → {msg}")


LEFT_TEST_JOINTS = [-0.35, 1.00, -0.35, 0.15, 0.0, 0.0]
RIGHT_TEST_JOINTS = [0.35, 1.00, -0.35, 0.15, 0.0, 0.0]
LEFT_GRASP_JOINTS = [-0.35, 1.45, -0.55, 0.25, 0.0, 0.0]
LEFT_PLACE_JOINTS = [0.35, 1.45, -0.55, 0.25, 0.0, 0.0]
RIGHT_GRASP_JOINTS = [0.35, 1.45, -0.55, 0.25, 0.0, 0.0]


def _pose_from_fk(fk_source, joint_angles: list, arm: str) -> dict:
    """
    用当前 URDF 的 FK 生成已知可达目标位姿。

    这样测试验证的是 IK/规划链路本身，而不是旧机器人尺寸下写死的坐标。
    """
    pose = fk_source.forward_kinematics(joint_angles, arm)
    return {
        "position": [float(v) for v in pose["position"]],
        "quaternion": [float(v) for v in pose["quaternion"]],
    }


# ============================================================
# 测试 1：坐标转换精度
# ============================================================

def test_coordinate_transform():
    """
    测试坐标转换模块。

    验证内容：
      - 内参反投影一致性（像素→3D→像素）
      - 俯视相机变换
      - 手腕相机变换（依赖关节角度）
      - 旋转矩阵正交性
      - 批量转换
    """
    _header("测试 1：坐标转换精度 (test_coordinate_transform)")

    from leisaac.motion_planning.coordinate_transform import (
        CameraToBaseTransformer,
        euler_to_rotation_matrix,
    )

    transformer = CameraToBaseTransformer()
    passed = 0
    total = 0

    # ---- 1.1 内参反投影一致性 ----
    _sub("1.1 内参反投影一致性")
    total += 1
    intr = transformer.intrinsics
    u_orig, v_orig, depth = 200.0, 150.0, 0.5
    p_cam = intr.pixel_to_camera(u_orig, v_orig, depth)
    u_back, v_back = intr.camera_to_pixel(p_cam)
    pixel_error = abs(u_orig - u_back) + abs(v_orig - v_back)
    _info(f"原始像素: ({u_orig}, {v_orig})")
    _info(f"相机坐标: ({p_cam[0]:.4f}, {p_cam[1]:.4f}, {p_cam[2]:.4f})")
    _info(f"反投影像素: ({u_back:.2f}, {v_back:.2f})")
    _info(f"像素误差: {pixel_error:.6f}")
    assert pixel_error < 0.01, f"反投影误差过大: {pixel_error}"
    _ok("反投影一致性 PASS")
    passed += 1

    # ---- 1.2 俯视相机变换 ----
    _sub("1.2 俯视相机变换")
    total += 1
    p_base = transformer.transform_top_camera(320, 240, 0.5)
    _info(f"像素 (320, 240) + 深度 0.5m → 基座坐标: ({p_base[0]:.4f}, {p_base[1]:.4f}, {p_base[2]:.4f})")
    assert np.all(np.isfinite(p_base)), "俯视相机变换产生非有限值"
    _ok("俯视相机变换 PASS")
    passed += 1

    # ---- 1.3 手腕相机变换 ----
    _sub("1.3 手腕相机变换")
    total += 1
    joints_a = np.array([0.0, -0.5, 0.3, 0.0, 0.0, 0.0])
    joints_b = np.array([0.5, -0.3, 0.5, 0.0, 0.0, 0.0])
    p_a = transformer.transform_wrist_camera(320, 240, 0.3, joints_a, "left")
    p_b = transformer.transform_wrist_camera(320, 240, 0.3, joints_b, "left")
    diff = np.linalg.norm(p_a - p_b)
    _info(f"关节 A 基座坐标: ({p_a[0]:.4f}, {p_a[1]:.4f}, {p_a[2]:.4f})")
    _info(f"关节 B 基座坐标: ({p_b[0]:.4f}, {p_b[1]:.4f}, {p_b[2]:.4f})")
    _info(f"差异: {diff:.4f}m")
    assert diff > 0.001, "不同关节角度应产生不同结果"
    assert np.all(np.isfinite(p_a)) and np.all(np.isfinite(p_b)), "手腕相机变换产生非有限值"
    _ok("手腕相机变换 PASS")
    passed += 1

    # ---- 1.4 旋转矩阵正交性 ----
    _sub("1.4 旋转矩阵正交性")
    total += 1
    R = euler_to_rotation_matrix(0.3, -0.5, 0.7)
    ortho_error = np.linalg.norm(R @ R.T - np.eye(3))
    _info(f"R*R' 与单位矩阵误差: {ortho_error:.2e}")
    assert ortho_error < 1e-10, f"旋转矩阵不正交: {ortho_error}"
    _ok("旋转矩阵正交性 PASS")
    passed += 1

    # ---- 1.5 批量转换 ----
    _sub("1.5 批量转换")
    total += 1
    detections = [(100, 100, 0.3), (320, 240, 0.5), (500, 400, 0.4)]
    batch_result = transformer.batch_transform(detections, "top")
    _info(f"输入: {len(detections)} 个检测")
    _info(f"输出形状: {batch_result.shape}")
    assert batch_result.shape == (3, 3), f"批量输出形状错误: {batch_result.shape}"
    assert np.all(np.isfinite(batch_result)), "批量转换产生非有限值"
    _ok("批量转换 PASS")
    passed += 1

    print(f"\n  坐标转换测试: {passed}/{total} 通过")
    return passed == total


# ============================================================
# 测试 2：IK 求解 + FK 验证
# ============================================================

def test_ik_solver():
    """
    测试 IK 求解器。

    验证内容：
      - 求解器初始化
      - 左臂 IK 求解
      - 右臂 IK 求解
      - 双臂同时求解
      - 关节限位验证
      - IK→FK 一致性
    """
    _header("测试 2：IK 求解 (test_ik_solver)")

    from leisaac.motion_planning.ik_solver import DualArmIKSolver

    passed = 0
    total = 0

    # ---- 2.1 初始化 ----
    _sub("2.1 求解器初始化")
    total += 1
    solver = DualArmIKSolver(robot_config="xtrainer.yml", num_seeds=32)
    _ok("初始化 PASS")
    passed += 1

    # ---- 2.2 左臂 IK ----
    _sub("2.2 左臂 IK 求解")
    total += 1
    left_target = _pose_from_fk(solver, LEFT_TEST_JOINTS, "left")
    t0 = time.time()
    result_l = solver.solve_left_arm(left_target)
    elapsed = (time.time() - t0) * 1000
    _info(f"目标: pos={left_target['position']}")
    _info(f"成功: {result_l['success']}")
    _info(f"耗时: {elapsed:.1f}ms")
    if result_l["success"]:
        _info(f"关节角度: {[f'{a:.4f}' for a in result_l['joint_angles']]}")
        _info(f"位置误差: {result_l['position_error_mm']:.3f}mm")
    assert result_l["success"], f"左臂 IK 失败: {result_l['error_message']}"
    assert result_l["position_error_mm"] < 5.0, f"位置误差过大: {result_l['position_error_mm']}mm"
    _ok("左臂 IK PASS")
    passed += 1

    # ---- 2.3 右臂 IK ----
    _sub("2.3 右臂 IK 求解")
    total += 1
    right_target = _pose_from_fk(solver, RIGHT_TEST_JOINTS, "right")
    result_r = solver.solve_right_arm(right_target)
    _info(f"成功: {result_r['success']}")
    if result_r["success"]:
        _info(f"位置误差: {result_r['position_error_mm']:.3f}mm")
    assert result_r["success"], f"右臂 IK 失败: {result_r['error_message']}"
    _ok("右臂 IK PASS")
    passed += 1

    # ---- 2.4 双臂同时求解 ----
    _sub("2.4 双臂同时求解")
    total += 1
    t0 = time.time()
    result_both = solver.solve_both(left_target, right_target)
    elapsed = (time.time() - t0) * 1000
    _info(f"双臂成功: {result_both['both_success']}")
    _info(f"耗时: {elapsed:.1f}ms")
    assert result_both["both_success"], "双臂同时求解失败"
    _ok("双臂同时求解 PASS")
    passed += 1

    # ---- 2.5 关节限位验证 ----
    _sub("2.5 关节限位验证")
    total += 1
    valid = solver.check_ik_valid([0.1, -0.5, 0.3, 0.0, 0.0, 0.0], "left")
    invalid = solver.check_ik_valid([0.1, -4.0, 0.3, 0.0, 0.0, 0.0], "left")
    _info(f"合法角度: valid={valid['valid']}")
    _info(f"超限角度: valid={invalid['valid']}, violations={invalid['violations']}")
    assert valid["valid"], "合法角度被误判为超限"
    assert not invalid["valid"], "超限角度未被检测"
    _ok("关节限位验证 PASS")
    passed += 1

    # ---- 2.6 IK→FK 一致性 ----
    _sub("2.6 IK→FK 一致性验证")
    total += 1
    fk = solver.forward_kinematics(result_l["joint_angles"], "left")
    pos_error = np.linalg.norm(np.array(fk["position"]) - np.array(left_target["position"]))
    _info(f"IK 目标: {left_target['position']}")
    _info(f"FK 结果: {[f'{v:.4f}' for v in fk['position']]}")
    _info(f"FK 位置误差: {pos_error*1000:.3f}mm")
    assert pos_error < 0.01, f"IK→FK 误差过大: {pos_error*1000:.3f}mm"
    _ok("IK→FK 一致性 PASS")
    passed += 1

    print(f"\n  IK 求解测试: {passed}/{total} 通过")
    return passed == total


# ============================================================
# 测试 3：轨迹规划 + 平滑性检查
# ============================================================

def test_motion_planner():
    """
    测试运动规划器。

    验证内容：
      - 规划器初始化 + 场景构建
      - plan_to_pose 左臂/右臂
      - plan_grasp 三阶段
      - 轨迹平滑性（相邻点关节角变化量）
      - 关节限位合规
    """
    _header("测试 3：轨迹规划 (test_motion_planner)")

    from leisaac.motion_planning.motion_planner import DualArmMotionPlanner

    passed = 0
    total = 0

    # ---- 3.1 初始化 + 场景 ----
    _sub("3.1 初始化 + 场景构建")
    total += 1
    planner = DualArmMotionPlanner(robot_config="xtrainer.yml")
    builder = planner.get_scene_builder()
    builder.add_table("桌子", [0.4, 0, 0.15], [0.8, 0.6, 0.03])
    builder.add_box("障碍盒", [0.3, 0.1, 0.25], [0.05, 0.05, 0.1])
    planner.apply_scene()
    planner.warmup()
    _ok("初始化 + 场景 + 预热 PASS")
    passed += 1

    # ---- 3.2 左臂 plan_to_pose ----
    _sub("3.2 左臂 plan_to_pose")
    total += 1
    left_target = _pose_from_fk(planner, LEFT_TEST_JOINTS, "left")
    t0 = time.time()
    result_l = planner.plan_to_pose("left", left_target, current_joints=[0]*6)
    elapsed = (time.time() - t0) * 1000
    _info(f"成功: {result_l['success']}")
    _info(f"轨迹点: {result_l['n_waypoints']}")
    _info(f"时长: {result_l['duration']:.2f}s")
    _info(f"耗时: {elapsed:.1f}ms")
    assert result_l["success"], f"左臂规划失败: {result_l['error_message']}"
    assert result_l["n_waypoints"] > 0, "轨迹点数为 0"
    _ok("左臂 plan_to_pose PASS")
    passed += 1

    # ---- 3.3 右臂 plan_to_pose ----
    _sub("3.3 右臂 plan_to_pose")
    total += 1
    right_target = _pose_from_fk(planner, RIGHT_TEST_JOINTS, "right")
    result_r = planner.plan_to_pose("right", right_target, current_joints=[0]*6)
    assert result_r["success"], f"右臂规划失败: {result_r['error_message']}"
    _ok("右臂 plan_to_pose PASS")
    passed += 1

    # ---- 3.4 轨迹平滑性检查 ----
    _sub("3.4 轨迹平滑性检查")
    total += 1
    traj = np.array(result_l["trajectory"])  # (N, 6)
    # 计算相邻点的最大关节角变化
    diffs = np.abs(np.diff(traj, axis=0))  # (N-1, 6)
    max_diff = np.max(diffs)
    mean_diff = np.mean(diffs)
    _info(f"轨迹形状: {traj.shape}")
    _info(f"相邻点最大关节变化: {max_diff:.4f} rad ({math.degrees(max_diff):.2f}°)")
    _info(f"相邻点平均关节变化: {mean_diff:.4f} rad ({math.degrees(mean_diff):.2f}°)")
    # 平滑性标准：相邻点变化不应超过 0.5 rad（约 28°）
    assert max_diff < 0.5, f"轨迹不平滑，最大跳变: {max_diff:.4f} rad"
    _ok("轨迹平滑性 PASS")
    passed += 1

    # ---- 3.5 关节限位合规 ----
    _sub("3.5 关节限位合规")
    total += 1
    within_limits = np.all(traj >= -np.pi) and np.all(traj <= np.pi)
    _info(f"全部在 [-π, π] 内: {within_limits}")
    assert within_limits, "轨迹中存在超出关节限位的点"
    _ok("关节限位合规 PASS")
    passed += 1

    # ---- 3.6 plan_grasp 三阶段 ----
    _sub("3.6 plan_grasp 三阶段抓取规划")
    total += 1
    grasp_pose = _pose_from_fk(planner, LEFT_GRASP_JOINTS, "left")
    result_g = planner.plan_grasp(
        "left", grasp_pose, current_joints=[0]*6, approach_offset=0.03, lift_offset=0.03
    )
    if result_g["success"]:
        _info(f"接近段: {len(result_g['approach_trajectory'])} 点")
        _info(f"抓取段: {len(result_g['grasp_trajectory'])} 点")
        _info(f"提升段: {len(result_g['lift_trajectory'])} 点")
        _info(f"总时长: {result_g['total_duration']:.2f}s")
    else:
        _info(f"抓取规划失败: {result_g['error_message']}")
    assert result_g["success"], f"三阶段抓取规划失败: {result_g['error_message']}"
    _ok("plan_grasp PASS")
    passed += 1

    print(f"\n  轨迹规划测试: {passed}/{total} 通过")
    return passed == total


# ============================================================
# 测试 4：状态机流转
# ============================================================

def test_state_machine():
    """
    测试任务状态机。

    验证内容：
      - 中间点生成
      - 单臂状态机完整流程
      - 双臂协调器
      - 错误处理 + 重试
      - 边界情况
    """
    _header("测试 4：状态机流转 (test_state_machine)")

    from leisaac.motion_planning.task_state_machine import (
        GraspTaskStateMachine, DualArmCoordinator,
        WaypointManager, Pose, TaskConfig, TaskState,
    )

    passed = 0
    total = 0

    config = TaskConfig(state_timeout=5.0, grasp_settle_time=0.01, release_settle_time=0.01)
    grasp_pose = Pose(np.array([0.3, 0.0, 0.1]), np.array([1.0, 0.0, 0.0, 0.0]))
    place_pose = Pose(np.array([0.3, 0.3, 0.1]), np.array([1.0, 0.0, 0.0, 0.0]))

    # ---- 4.1 中间点生成 ----
    _sub("4.1 中间点生成")
    total += 1
    wm = WaypointManager()
    wm.generate_grasp_waypoints(grasp_pose, place_pose, config)
    _info(f"中间点数量: {wm.total_count} (期望: 8)")
    assert wm.total_count == 8, f"中间点数量错误: {wm.total_count}"
    for i, wp in enumerate(wm.get_all_waypoints()):
        _info(f"  [{i}] {wp.description} | gripper={wp.gripper_state} speed={wp.speed_factor}")
    _ok("中间点生成 PASS")
    passed += 1

    # ---- 4.2 单臂状态机完整流程 ----
    _sub("4.2 单臂状态机完整流程")
    total += 1
    import time as time_module
    fsm = GraspTaskStateMachine("left", config)
    assert fsm.is_idle(), "初始状态应为 IDLE"
    fsm.start(grasp_pose, place_pose)
    _info(f"启动后状态: {fsm.state_name}")

    step = 0
    while not fsm.is_done() and not fsm.is_error() and step < 50:
        fsm.update()
        if fsm.state in (TaskState.MOVE_ABOVE, TaskState.DESCEND,
                         TaskState.LIFT, TaskState.TRANSIT,
                         TaskState.PLACE_DESCEND, TaskState.RETREAT):
            fsm.signal_arrived()
        step += 1
        time_module.sleep(0.02)

    _info(f"最终状态: {fsm.state_name}")
    _info(f"状态转换次数: {len(fsm.get_history())}")
    _info(f"执行步骤数: {step}")
    assert fsm.is_done(), f"状态机未完成，当前状态: {fsm.state_name}"
    _ok("单臂状态机完整流程 PASS")
    passed += 1

    # ---- 4.3 双臂协调器 ----
    _sub("4.3 双臂协调器")
    total += 1
    coordinator = DualArmCoordinator(config)
    left_g = Pose(np.array([0.3, -0.2, 0.1]), np.array([1, 0, 0, 0]))
    left_p = Pose(np.array([0.3, -0.2, 0.3]), np.array([1, 0, 0, 0]))
    right_g = Pose(np.array([0.3, 0.2, 0.1]), np.array([1, 0, 0, 0]))
    right_p = Pose(np.array([0.3, 0.2, 0.3]), np.array([1, 0, 0, 0]))
    coordinator.start_both(left_g, left_p, right_g, right_p)

    for _ in range(50):
        coordinator.update_all()
        for arm_fsm in (coordinator.left, coordinator.right):
            if arm_fsm.state in (TaskState.MOVE_ABOVE, TaskState.DESCEND,
                                 TaskState.LIFT, TaskState.TRANSIT,
                                 TaskState.PLACE_DESCEND, TaskState.RETREAT):
                arm_fsm.signal_arrived()
        time_module.sleep(0.02)

    status = coordinator.get_status()
    _info(f"左臂: state={status['left']['state']}, done={status['left']['is_done']}")
    _info(f"右臂: state={status['right']['state']}, done={status['right']['is_done']}")
    assert coordinator.is_all_done(), "双臂协调器未全部完成"
    _ok("双臂协调器 PASS")
    passed += 1

    # ---- 4.4 错误处理 + 重试 ----
    _sub("4.4 错误处理 + 重试")
    total += 1
    fsm_err = GraspTaskStateMachine("right", config)
    fsm_err.start(grasp_pose, place_pose)
    fsm_err.signal_error("IK 求解失败")
    assert fsm_err.is_error(), "应处于错误状态"
    _info(f"错误状态: {fsm_err.state_name}, msg={fsm_err.get_error_message()}")

    fsm_err.retry()
    assert fsm_err.state == TaskState.MOVE_ABOVE, f"重试后应为 MOVE_ABOVE，实际: {fsm_err.state_name}"
    _info(f"重试后状态: {fsm_err.state_name}")
    _ok("错误处理 + 重试 PASS")
    passed += 1

    # ---- 4.5 边界情况 ----
    _sub("4.5 边界情况")
    total += 1
    wm_empty = WaypointManager()
    assert not wm_empty.has_next() and wm_empty.get_next_waypoint() is None, "空管理器应返回 None"
    _ok("空管理器 PASS")

    wm_single = WaypointManager()
    wm_single.add_waypoint(grasp_pose, description="单个点")
    assert wm_single.total_count == 1 and wm_single.has_next(), "单点管理器应有 1 个点"
    wp = wm_single.get_next_waypoint()
    assert wp is not None and wp.description == "单个点" and not wm_single.has_next(), "取出后应为空"
    _ok("单点管理器 PASS")
    _ok("边界情况 PASS")
    passed += 1

    print(f"\n  状态机测试: {passed}/{total} 通过")
    return passed == total


# ============================================================
# 测试 5：完整流水线
# ============================================================

def test_full_pipeline():
    """
    完整流水线测试：目标位姿 → IK → 避障轨迹 → 关节轨迹。

    模拟 policy_server 的完整调用链：
      1. 构建场景
      2. 接收目标位姿
      3. IK 求解
      4. 避障轨迹规划
      5. 返回可执行的关节轨迹
    """
    _header("测试 5：完整流水线 (test_full_pipeline)")

    from leisaac.motion_planning.planner_interface import MotionPlanningModule

    passed = 0
    total = 0

    # ---- 5.1 初始化 ----
    _sub("5.1 初始化模块 + 预热")
    total += 1
    module = MotionPlanningModule()
    module.warmup()
    _ok("初始化 + 预热 PASS")
    passed += 1

    # ---- 5.2 设置场景 ----
    _sub("5.2 设置比赛场景")
    total += 1
    obstacles = [
        {"name": "桌子", "position": [0.4, 0, 0.15], "dimensions": [0.8, 0.6, 0.03]},
        {"name": "左挡板", "position": [0, -0.4, 0.3], "dimensions": [1.0, 0.02, 0.6]},
        {"name": "右挡板", "position": [0, 0.4, 0.3], "dimensions": [1.0, 0.02, 0.6]},
        {"name": "前挡板", "position": [0.7, 0, 0.3], "dimensions": [0.02, 0.8, 0.6]},
    ]
    module.update_scene(obstacles)
    _info(f"障碍物数量: {len(obstacles)}")
    _ok("场景设置 PASS")
    passed += 1

    # ---- 5.3 左臂完整规划 ----
    _sub("5.3 左臂 plan() 完整规划")
    total += 1
    target_left = _pose_from_fk(module._ik, LEFT_TEST_JOINTS, "left")
    t0 = time.time()
    result = module.plan(target_left, obstacles=obstacles, arm="left")
    elapsed = (time.time() - t0) * 1000
    _info(f"成功: {result.success}")
    _info(f"轨迹点: {result.n_waypoints}")
    _info(f"时长: {result.duration:.2f}s")
    _info(f"耗时: {elapsed:.1f}ms")
    if result.target_joints is not None:
        _info(f"目标关节: {[f'{a:.4f}' for a in result.target_joints]}")
    else:
        _info("目标关节: None（规划未成功生成目标关节）")
    assert result.success, f"左臂规划失败: {result.error_message}"
    assert result.n_waypoints > 0, "轨迹点数为 0"
    _ok("左臂 plan() PASS")
    passed += 1

    # ---- 5.4 右臂完整规划 ----
    _sub("5.4 右臂 plan() 完整规划")
    total += 1
    target_right = _pose_from_fk(module._ik, RIGHT_TEST_JOINTS, "right")
    result_r = module.plan(target_right, obstacles=obstacles, arm="right")
    assert result_r.success, f"右臂规划失败: {result_r.error_message}"
    _ok("右臂 plan() PASS")
    passed += 1

    # ---- 5.5 JSON 序列化 ----
    _sub("5.5 JSON 序列化验证")
    total += 1
    json_str = result.to_json()
    import json
    restored = json.loads(json_str)
    assert restored["success"] == result.success, "JSON 反序列化 success 不一致"
    assert restored["arm"] == result.arm, "JSON 反序列化 arm 不一致"
    assert len(restored["trajectory"]) == result.n_waypoints, "JSON 反序列化轨迹点数不一致"
    _info(f"JSON 长度: {len(json_str)} 字符")
    _info(f"轨迹点数: {len(restored['trajectory'])}")
    _ok("JSON 序列化 PASS")
    passed += 1

    # ---- 5.6 抓取序列规划 ----
    _sub("5.6 plan_grasp_sequence() 完整抓取序列")
    total += 1
    grasp_pose = _pose_from_fk(module._ik, LEFT_GRASP_JOINTS, "left")
    place_pose = _pose_from_fk(module._ik, LEFT_PLACE_JOINTS, "left")
    t0 = time.time()
    result_seq = module.plan_grasp_sequence(grasp_pose, place_pose, obstacles, arm="left")
    elapsed = (time.time() - t0) * 1000
    _info(f"成功: {result_seq.success}")
    _info(f"总轨迹点: {result_seq.n_waypoints}")
    _info(f"总时长: {result_seq.duration:.2f}s")
    _info(f"耗时: {elapsed:.1f}ms")
    if not result_seq.success:
        _info(f"失败阶段: {result_seq.phase}")
        _info(f"错误信息: {result_seq.error_message}")
    assert result_seq.success, f"抓取序列规划失败: {result_seq.error_message}"
    assert result_seq.n_waypoints > 0, "抓取序列轨迹点数为 0"
    _ok("plan_grasp_sequence() PASS")
    passed += 1

    # ---- 5.7 状态同步 ----
    _sub("5.7 状态同步验证")
    total += 1
    state = module.get_current_state()
    _info(f"左臂关节: {[f'{a:.4f}' for a in state['left']]}")
    _info(f"右臂关节: {[f'{a:.4f}' for a in state['right']]}")
    assert len(state["left"]) == 6, "左臂应有 6 个关节"
    assert len(state["right"]) == 6, "右臂应有 6 个关节"
    _ok("状态同步 PASS")
    passed += 1

    # ---- 5.8 异常处理 ----
    _sub("5.8 异常处理验证")
    total += 1
    # 超出工作空间
    bad = module.plan({"position": [2.0, 0, 0], "quaternion": [1, 0, 0, 0]}, arm="left")
    assert not bad.success, "超远目标应失败"
    assert "工作空间" in bad.error_message, f"错误信息应提及工作空间: {bad.error_message}"
    _info(f"超远目标: success={bad.success}, msg={bad.error_message[:50]}")
    # 缺少字段
    bad2 = module.plan({"position": [0.3, 0, 0.3]}, arm="left")
    assert not bad2.success, "缺少字段应失败"
    _info(f"缺少字段: success={bad2.success}, msg={bad2.error_message[:50]}")
    _ok("异常处理 PASS")
    passed += 1

    print(f"\n  完整流水线测试: {passed}/{total} 通过")
    return passed == total


# ============================================================
# 一键运行所有测试
# ============================================================

def run_all_tests():
    """
    一键运行所有测试。

    返回：
        bool: 全部通过返回 True
    """
    print("\n" + "#" * 60)
    print("  运动规划模块 —— 全量测试")
    print("#" * 60)

    results = {}
    t_total = time.time()

    # 运行每个测试（捕获异常，避免一个失败阻塞全部）
    for name, func in [
        ("坐标转换", test_coordinate_transform),
        ("IK 求解", test_ik_solver),
        ("轨迹规划", test_motion_planner),
        ("状态机", test_state_machine),
        ("完整流水线", test_full_pipeline),
    ]:
        try:
            results[name] = func()
        except Exception as e:
            _fail(f"{name} 测试异常: {e}")
            results[name] = False

    # ---- 汇总 ----
    elapsed_total = time.time() - t_total
    print("\n" + "=" * 60)
    print("  测试汇总")
    print("=" * 60)
    all_pass = True
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  {status}  {name}")

    print(f"\n  总耗时: {elapsed_total:.2f}s")
    if all_pass:
        print("\n  [ALL PASS] 全部测试通过！运动规划模块就绪。")
    else:
        print("\n  [SOME FAIL] 部分测试失败，请检查上方日志。")
    print("=" * 60)

    return all_pass


# ============================================================
# 主函数入口
# ============================================================

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
