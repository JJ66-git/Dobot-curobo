"""
为 Dobot X-Trainer 双臂机器人生成 cuRobo 配置文件（YAML）
==========================================================

功能：
  1. 解析 xtrainer.urdf 的运动学结构
  2. 为每个连杆生成碰撞球（基于连杆几何形状自动计算）
  3. 计算自碰撞忽略矩阵（相邻连杆对自动忽略）
  4. 导出 cuRobo 兼容的 YAML 配置文件

使用方式：
  cd C:\\jj\\Dobot\\Guangdong_Collegiate_Computing_Competition
  python curobo/my_x_trainer/build_model.py

输出文件：
  curobo/my_x_trainer/xtrainer.yml
"""

import sys
import math
import xml.etree.ElementTree as ET
from pathlib import Path

# PyYAML：用于写入 YAML 格式配置文件
import yaml

# ============================================================
# 路径常量
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent          # curobo/my_x_trainer/
URDF_PATH = SCRIPT_DIR / "xtrainer.urdf"
OUTPUT_PATH = SCRIPT_DIR / "xtrainer.yml"
TOOL_FRAMES = ["left_ee_link", "right_ee_link"]


# ============================================================
# 第一步：解析 URDF，提取连杆和关节信息
# ============================================================

def parse_urdf(urdf_path: str):
    """
    解析 URDF 文件，提取所有连杆名和关节名。

    参数：
        urdf_path: URDF 文件路径

    返回：
        tuple: (link_names列表, joint_names列表, revolute_joint_names列表)
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    link_names = [link.attrib["name"] for link in root.findall("link")]
    joints = root.findall("joint")
    joint_names = [j.attrib["name"] for j in joints]
    revolute_names = [j.attrib["name"] for j in joints if j.attrib["type"] == "revolute"]

    return link_names, joint_names, revolute_names


# ============================================================
# 第二步：碰撞球生成函数
# ============================================================

def generate_cylinder_spheres(length, radius, n_length=3, n_angular=4):
    """
    为圆柱形连杆生成碰撞球。
    沿轴线均匀放置 n_length 层球体，每层在圆周上放 n_angular 个球。
    """
    spheres = []
    sphere_radius = radius * 1.1  # 比圆柱半径大 10%，确保覆盖

    for i in range(n_length):
        z = length * (i + 0.5) / n_length if n_length > 1 else length / 2.0
        if n_angular == 1:
            spheres.append({"center": [0.0, 0.0, z], "radius": sphere_radius})
        else:
            for j in range(n_angular):
                angle = 2.0 * math.pi * j / n_angular
                offset = radius * 0.5
                spheres.append({
                    "center": [offset * math.cos(angle), offset * math.sin(angle), z],
                    "radius": sphere_radius,
                })
    return spheres


def generate_box_spheres(dx, dy, dz, n_per_axis=2):
    """
    为长方体连杆生成碰撞球。
    在长方体内部均匀放置球体，球半径足以覆盖到最近表面。
    """
    spheres = []
    cell_dx, cell_dy, cell_dz = dx / n_per_axis, dy / n_per_axis, dz / n_per_axis
    sphere_radius = math.sqrt(cell_dx**2 + cell_dy**2 + cell_dz**2) / 2.0 * 1.1

    for ix in range(n_per_axis):
        for iy in range(n_per_axis):
            for iz in range(n_per_axis):
                spheres.append({
                    "center": [
                        (ix + 0.5) * cell_dx - dx / 2.0,
                        (iy + 0.5) * cell_dy - dy / 2.0,
                        (iz + 0.5) * cell_dz,
                    ],
                    "radius": sphere_radius,
                })
    return spheres


# ============================================================
# 第三步：定义每个连杆的碰撞球参数
# ============================================================
# 基于 URDF 中的连杆尺寸估算
# 格式：link_name -> (形状类型, 参数...)
#   "cylinder": (长度, 半径, 沿轴球数, 圆周球数)
#   "box": (dx, dy, dz, 每轴球数)

LINK_SPHERE_PARAMS = {
    "base_link":       ("box",      0.12, 0.12, 0.06, 2),
    "left_arm_base":   ("cylinder", 0.06, 0.04, 2, 3),
    "right_arm_base":  ("cylinder", 0.06, 0.04, 2, 3),
    "J1_1_link":       ("cylinder", 0.05, 0.035, 2, 3),
    "J1_2_link":       ("cylinder", 0.11, 0.030, 3, 3),
    "J1_3_link":       ("cylinder", 0.10, 0.025, 3, 3),
    "J1_4_link":       ("cylinder", 0.04, 0.020, 2, 3),
    "J1_5_link":       ("cylinder", 0.035, 0.018, 2, 3),
    "J1_6_link":       ("cylinder", 0.04, 0.015, 2, 3),
    "J1_7_link":       ("cylinder", 0.04, 0.010, 2, 2),
    "J1_8_link":       ("cylinder", 0.04, 0.010, 2, 2),
    "J2_1_link":       ("cylinder", 0.05, 0.035, 2, 3),
    "J2_2_link":       ("cylinder", 0.11, 0.030, 3, 3),
    "J2_3_link":       ("cylinder", 0.10, 0.025, 3, 3),
    "J2_4_link":       ("cylinder", 0.04, 0.020, 2, 3),
    "J2_5_link":       ("cylinder", 0.035, 0.018, 2, 3),
    "J2_6_link":       ("cylinder", 0.04, 0.015, 2, 3),
    "J2_7_link":       ("cylinder", 0.04, 0.010, 2, 2),
    "J2_8_link":       ("cylinder", 0.04, 0.010, 2, 2),
}


def generate_all_collision_spheres():
    """
    为所有连杆生成碰撞球。

    返回：
        dict: {link_name: [{"center": [x,y,z], "radius": r}, ...], ...}
    """
    all_spheres = {}
    total = 0

    for link_name, params in LINK_SPHERE_PARAMS.items():
        shape = params[0]
        if shape == "cylinder":
            spheres = generate_cylinder_spheres(*params[1:])
        elif shape == "box":
            spheres = generate_box_spheres(*params[1:])
        else:
            raise ValueError(f"未知形状: {shape}")

        all_spheres[link_name] = spheres
        total += len(spheres)
        print(f"  {link_name:<20s}: {len(spheres)} 个碰撞球 ({shape})")

    print(f"\n  总计: {total} 个碰撞球，覆盖 {len(all_spheres)} 个连杆")
    return all_spheres


# ============================================================
# 第四步：自碰撞忽略矩阵
# ============================================================
# 相邻连杆永远在碰撞范围内，必须忽略，否则碰撞检测永远报错

SELF_COLLISION_IGNORE = {
    "J1_1_link": ["left_arm_base", "J1_2_link"],
    "J1_2_link": ["J1_1_link", "J1_3_link"],
    "J1_3_link": ["J1_2_link", "J1_4_link"],
    "J1_4_link": ["J1_3_link", "J1_5_link"],
    "J1_5_link": ["J1_4_link", "J1_6_link"],
    "J1_6_link": ["J1_5_link", "J1_7_link", "J1_8_link", "left_ee_link"],
    "J1_7_link": ["J1_6_link", "J1_8_link"],
    "J1_8_link": ["J1_6_link", "J1_7_link"],
    "J2_1_link": ["right_arm_base", "J2_2_link"],
    "J2_2_link": ["J2_1_link", "J2_3_link"],
    "J2_3_link": ["J2_2_link", "J2_4_link"],
    "J2_4_link": ["J2_3_link", "J2_5_link"],
    "J2_5_link": ["J2_4_link", "J2_6_link"],
    "J2_6_link": ["J2_5_link", "J2_7_link", "J2_8_link", "right_ee_link"],
    "J2_7_link": ["J2_6_link", "J2_8_link"],
    "J2_8_link": ["J2_6_link", "J2_7_link"],
    "base_link": ["left_arm_base", "right_arm_base", "world_base_link"],
    "left_arm_base": ["base_link", "J1_1_link"],
    "right_arm_base": ["base_link", "J2_1_link"],
}

SELF_COLLISION_BUFFER = {
    "base_link": 0.02,
    "left_arm_base": 0.02,
    "right_arm_base": 0.02,
}


# ============================================================
# 第五步：组装并写入 YAML 配置文件
# ============================================================

def build_and_save_yaml(urdf_path: str, tool_frames: list, output_path: str):
    """
    生成完整的 cuRobo YAML 配置文件并保存。

    参数：
        urdf_path: URDF 文件路径
        tool_frames: 末端执行器名称列表
        output_path: 输出 YAML 文件路径
    """
    print("\n" + "=" * 60)
    print("生成 cuRobo 配置文件")
    print("=" * 60)

    # A. 解析 URDF
    print("\n[A] 解析 URDF 文件...")
    link_names, joint_names, revolute_names = parse_urdf(urdf_path)
    print(f"  连杆数: {len(link_names)}")
    print(f"  关节数: {len(joint_names)} (旋转: {len(revolute_names)})")
    print(f"  旋转关节: {revolute_names}")

    # B. 生成碰撞球
    print("\n[B] 生成碰撞球...")
    collision_spheres = generate_all_collision_spheres()

    # C. 关节空间配置
    print("\n[C] 配置关节空间...")
    # 只有旋转关节参与 IK/规划
    cspace_config = {
        "joint_names": revolute_names,
        "default_joint_position": [0.0] * len(revolute_names),
        "null_space_weight": [1.0] * len(revolute_names),
        "cspace_distance_weight": [1.0] * len(revolute_names),
        "max_jerk": 500.0,
        "max_acceleration": 15.0,
    }
    print(f"  活跃关节: {len(revolute_names)} 个 -> {revolute_names}")

    # D. 组装数据结构
    # 注意：cuRobo YAML 的顶层 key 是 "robot_cfg"，其下是 "kinematics"
    print("\n[D] 组装配置数据...")
    urdf_filename = Path(urdf_path).name
    asset_root = "."

    data_dict = {
        "robot_cfg": {
            "kinematics": {
                "urdf_path": urdf_filename,
                "asset_root_path": asset_root,
                "base_link": "world_base_link",
                "tool_frames": tool_frames,
                "collision_link_names": list(collision_spheres.keys()),
                "collision_spheres": collision_spheres,
                "self_collision_ignore": SELF_COLLISION_IGNORE,
                "self_collision_buffer": SELF_COLLISION_BUFFER,
                "mesh_link_names": list(collision_spheres.keys()),
                "cspace": cspace_config,
                "format_version": 2.0,
            }
        }
    }

    # E. 写入 YAML
    print(f"\n[E] 写入 YAML: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # 使用 yaml.dump 写入，default_flow_style=False 让列表/字典以块格式显示
    # allow_unicode=True 支持中文注释（如果有的话）
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(
            data_dict,
            f,
            default_flow_style=False,   # 块格式（更易读）
            allow_unicode=True,          # 支持中文
            sort_keys=False,             # 保持插入顺序
            width=120,                   # 行宽
        )

    file_size = Path(output_path).stat().st_size
    print(f"  文件大小: {file_size / 1024:.1f} KB")

    return data_dict


# ============================================================
# 第六步：验证生成的 YAML（基础结构检查，不依赖 cuRobo）
# ============================================================

def verify_yaml_structure(yaml_path: str, tool_frames: list, revolute_names: list):
    """
    验证生成的 YAML 文件结构是否正确（不依赖 cuRobo）。

    参数：
        yaml_path: YAML 文件路径
        tool_frames: 期望的末端执行器名
        revolute_names: 期望的旋转关节名
    """
    print("\n" + "=" * 60)
    print("验证 YAML 结构")
    print("=" * 60)

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    checks = []

    # 检查 1: 顶层结构
    has_robot_cfg = "robot_cfg" in data
    checks.append(("robot_cfg 顶层 key", has_robot_cfg))

    if not has_robot_cfg:
        print("[错误] YAML 缺少 robot_cfg 顶层 key")
        return False

    kin = data["robot_cfg"]["kinematics"]

    # 检查 2: 必要字段
    checks.append(("urdf_path 字段", "urdf_path" in kin))
    checks.append(("base_link 字段", "base_link" in kin))
    checks.append(("tool_frames 字段", "tool_frames" in kin))
    checks.append(("collision_spheres 字段", "collision_spheres" in kin))
    checks.append(("self_collision_ignore 字段", "self_collision_ignore" in kin))
    checks.append(("cspace 字段", "cspace" in kin))

    # 检查 3: tool_frames 正确
    checks.append(("tool_frames 内容", set(kin.get("tool_frames", [])) == set(tool_frames)))

    # 检查 4: cspace joint_names 正确
    cspace_joints = kin.get("cspace", {}).get("joint_names", [])
    checks.append(("cspace 关节数量", len(cspace_joints) == len(revolute_names)))
    checks.append(("cspace 关节名匹配", set(cspace_joints) == set(revolute_names)))

    # 检查 5: collision_spheres 非空
    n_spheres = sum(len(v) for v in kin.get("collision_spheres", {}).values())
    checks.append(("碰撞球非空", n_spheres > 0))

    # 打印结果
    all_pass = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}")

    print(f"\n  碰撞球总数: {n_spheres}")
    print(f"  关节数量: {len(cspace_joints)}")

    if all_pass:
        print("\n[成功] YAML 结构验证全部通过！")
    else:
        print("\n[警告] 部分验证未通过，请检查 YAML 文件")

    return all_pass


# ============================================================
# 主函数
# ============================================================

def main():
    """
    主流程：生成 cuRobo 配置文件并验证。
    """
    print("=" * 60)
    print("Dobot X-Trainer cuRobo 配置文件生成器")
    print("=" * 60)

    # 路径
    urdf_path = str(URDF_PATH)
    output_path = str(OUTPUT_PATH)

    print(f"\nURDF 文件: {urdf_path}")
    print(f"输出文件:  {output_path}")
    print(f"工具帧:    {TOOL_FRAMES}")

    # 检查 URDF 存在
    if not Path(urdf_path).exists():
        print(f"\n[错误] URDF 不存在: {urdf_path}")
        print("[信息] 请先创建 xtrainer.urdf (Step 1)")
        sys.exit(1)

    # 解析 URDF 获取关节名（验证用）
    _, _, revolute_names = parse_urdf(urdf_path)

    # 生成 YAML
    build_and_save_yaml(urdf_path, TOOL_FRAMES, output_path)

    # 验证
    verify_yaml_structure(output_path, TOOL_FRAMES, revolute_names)

    # 完成
    print("\n" + "=" * 60)
    print("完成！")
    print(f"  配置文件: {output_path}")
    print("  下一步: Step 3 (坐标转换) 或 Step 5 (IK 求解)")
    print("=" * 60)


if __name__ == "__main__":
    main()
