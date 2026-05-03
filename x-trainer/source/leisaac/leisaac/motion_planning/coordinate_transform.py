"""
相机坐标 → 机器人基座坐标 转换模块
====================================

功能：
  将相机检测到的目标位置（像素坐标 + 深度）转换为机器人基座坐标系下的 3D 坐标。

支持的相机：
  - left_wrist:  左腕手眼相机，安装在 J1_6 连杆上（需要当前关节角度）
  - right_wrist: 右腕手眼相机，安装在 J2_6 连杆上（需要当前关节角度）
  - top:         俯视相机，安装在 base_link 上（固定变换，无需关节角度）

坐标转换链：
  像素 (u, v, depth)
    → 相机坐标系 P_cam（通过相机内参反投影）
    → 基座坐标系 P_base（通过外参变换矩阵）

  手腕相机：P_base = T_base_ee @ T_ee_camera @ P_cam
  俯视相机：P_base = T_base_camera @ P_cam

依赖：仅 numpy
"""

import numpy as np
from typing import Dict, Optional, Tuple


# ============================================================
# 第一部分：旋转矩阵工具函数
# ============================================================

def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    欧拉角 → 旋转矩阵（ZYX 内旋约定）。

    欧拉角含义：
      roll  (绕 X 轴旋转)
      pitch (绕 Y 轴旋转)
      yaw   (绕 Z 轴旋转)

    旋转顺序：先绕 X (roll)，再绕 Y (pitch)，最后绕 Z (yaw)
    即 R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

    参数：
        roll, pitch, yaw: 弧度制角度

    返回：
        3x3 旋转矩阵
    """
    cr, sr = np.cos(roll), np.sin(roll)      # roll 的余弦和正弦
    cp, sp = np.cos(pitch), np.sin(pitch)    # pitch 的余弦和正弦
    cy, sy = np.cos(yaw), np.sin(yaw)        # yaw 的余弦和正弦

    # 绕 X 轴旋转矩阵
    Rx = np.array([
        [1,  0,   0 ],
        [0,  cr, -sr],
        [0,  sr,  cr],
    ])

    # 绕 Y 轴旋转矩阵
    Ry = np.array([
        [ cp, 0, sp],
        [ 0,  1, 0 ],
        [-sp, 0, cp],
    ])

    # 绕 Z 轴旋转矩阵
    Rz = np.array([
        [cy, -sy, 0],
        [sy,  cy, 0],
        [0,   0,  1],
    ])

    # ZYX 内旋：R = Rz @ Ry @ Rx
    return Rz @ Ry @ Rx


def euler_to_transform_matrix(roll: float, pitch: float, yaw: float,
                               translation: np.ndarray) -> np.ndarray:
    """
    欧拉角 + 平移 → 4x4 齐次变换矩阵。

    齐次变换矩阵的含义：
      描述一个坐标系相对于另一个坐标系的位姿（位置 + 姿态）。

      T = [R  t]    R: 3x3 旋转矩阵（描述朝向）
          [0  1]    t: 3x1 平移向量（描述位置）

    参数：
        roll, pitch, yaw: 弧度制欧拉角
        translation: [tx, ty, tz] 平移向量

    返回：
        4x4 齐次变换矩阵
    """
    R = euler_to_rotation_matrix(roll, pitch, yaw)
    T = np.eye(4)
    T[:3, :3] = R                   # 左上 3x3 = 旋转部分
    T[:3, 3] = translation          # 右上 3x1 = 平移部分
    return T


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    旋转矩阵 → 四元数 [qw, qx, qy, qz]（cuRobo 约定：wxyz 顺序）。

    使用 Shepperd 方法，避免万向节锁和数值不稳定。

    参数：
        R: 3x3 旋转矩阵

    返回：
        [qw, qx, qy, qz] 四元数
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]  # 旋转矩阵的迹

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return np.array([w, x, y, z])


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    四元数 [qw, qx, qy, qz] → 旋转矩阵。

    参数：
        q: [qw, qx, qy, qz] 四元数（wxyz 顺序）

    返回：
        3x3 旋转矩阵
    """
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)    ],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)    ],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def transform_from_pose(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    """
    位姿（位置 + 四元数）→ 4x4 齐次变换矩阵。

    参数：
        position: [x, y, z] 位置
        quaternion: [qw, qx, qy, qz] 四元数

    返回：
        4x4 齐次变换矩阵
    """
    R = quaternion_to_rotation_matrix(quaternion)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = position
    return T


# ============================================================
# 第二部分：相机内参
# ============================================================

class CameraIntrinsics:
    """
    相机内参类。

    相机内参描述了"像素坐标"和"相机坐标系中的 3D 坐标"之间的关系。

    关键公式（针孔相机模型）：
      u = fx * X / Z + cx
      v = fy * Y / Z + cy

    反投影（像素 → 3D 相机坐标）：
      X = (u - cx) * Z / fx
      Y = (v - cy) * Z / fy
      Z = depth（已知）

    参数来源（Isaac Sim TiledCamera 配置）：
      focal_length = 26.8 mm
      horizontal_aperture = 36.83 mm
      width = 640, height = 480
    """

    def __init__(self, focal_length: float, horizontal_aperture: float,
                 width: int, height: int):
        """
        参数：
            focal_length: 焦距（mm）
            horizontal_aperture: 水平光圈宽度（mm）
            width: 图像宽度（像素）
            height: 图像高度（像素）
        """
        # fx = 焦距(像素) = 焦距(mm) * 图像宽度(像素) / 水平光圈(mm)
        self.fx = focal_length * width / horizontal_aperture
        # fy = fy，假设像素是正方形（fy ≈ fx）
        self.fy = self.fx
        # 主点（光轴与成像平面的交点），通常在图像中心
        self.cx = width / 2.0
        self.cy = height / 2.0
        self.width = width
        self.height = height

    def pixel_to_camera(self, u: float, v: float, depth: float) -> np.ndarray:
        """
        像素坐标 + 深度 → 相机坐标系下的 3D 坐标。

        原理：
          针孔相机模型的反投影。已知像素位置 (u, v) 和该点到相机的距离 depth，
          计算该点在相机坐标系中的 3D 位置 (X, Y, Z)。

        公式：
          X = (u - cx) * depth / fx
          Y = (v - cy) * depth / fy
          Z = depth

        参数：
            u: 像素列坐标（水平方向，0~639）
            v: 像素行坐标（垂直方向，0~479）
            depth: 深度值（米），即目标到相机光心的距离

        返回：
            [X, Y, Z] 相机坐标系下的 3D 坐标（米）
        """
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        z = depth
        return np.array([x, y, z])

    def camera_to_pixel(self, point_cam: np.ndarray) -> Tuple[float, float]:
        """
        相机坐标系下的 3D 坐标 → 像素坐标（正投影，用于验证）。

        参数：
            point_cam: [X, Y, Z] 相机坐标系下的 3D 坐标

        返回：
            (u, v) 像素坐标
        """
        u = self.fx * point_cam[0] / point_cam[2] + self.cx
        v = self.fy * point_cam[1] / point_cam[2] + self.cy
        return (u, v)

    def __repr__(self):
        return (f"CameraIntrinsics(fx={self.fx:.1f}, fy={self.fy:.1f}, "
                f"cx={self.cx:.1f}, cy={self.cy:.1f}, "
                f"size={self.width}x{self.height})")


# ============================================================
# 第三部分：X-Trainer 正运动学（简化版，仅用于坐标转换）
# ============================================================

def compute_arm_fk(joint_angles: np.ndarray, arm: str = "left") -> np.ndarray:
    """
    简化版正运动学：从关节角度计算末端执行器的 4x4 变换矩阵。

    这是一个简化的 DH 参数模型，用于坐标转换模块。
    完整的运动学由 cuRobo 的 Kinematics 模块提供（Step 5）。

    X-Trainer 臂的 DH 参数（基于 URDF 连杆结构）：
      关节 1 (J_1): 肩旋转，绕 Z 轴，offset z = 0.00m（在臂基座处）
      关节 2 (J_2): 肩抬升，绕 Y 轴，offset z = 0.05m
      关节 3 (J_3): 肘弯曲，绕 Y 轴，offset z = 0.11m
      关节 4 (J_4): 腕弯曲，绕 Y 轴，offset z = 0.10m
      关节 5 (J_5): 腕旋转，绕 X 轴，offset z = 0.04m
      关节 6 (J_6): 腕末端，绕 Z 轴，offset z = 0.035m
      末端偏移: z = 0.04m（到 ee_link）

    参数：
        joint_angles: 6 个关节角度（弧度），顺序 [J1, J2, J3, J4, J5, J6]
        arm: "left" 或 "right"

    返回：
        4x4 齐次变换矩阵 T_base_ee（从基座到末端执行器）
    """
    assert len(joint_angles) == 6, f"需要 6 个关节角度，收到 {len(joint_angles)}"

    # 臂基座偏移（相对于 base_link）
    if arm == "left":
        arm_base_offset = np.array([-0.075, 0.0, 0.06])
    elif arm == "right":
        arm_base_offset = np.array([0.075, 0.0, 0.06])
    else:
        raise ValueError(f"arm 必须是 'left' 或 'right'，收到 '{arm}'")

    # DH 参数：(a, alpha, d, theta_offset)
    # a: 连杆长度（沿 X 轴）
    # alpha: 连杆扭角（绕 X 轴）
    # d: 连杆偏移（沿 Z 轴）
    # theta_offset: 关节角度偏移
    dh_params = [
        (0.0,   0.0,        0.0,   0.0),   # J1: 肩旋转（绕 Z）
        (0.0,   np.pi/2,    0.05,  0.0),   # J2: 肩抬升（绕 Y，经 alpha 旋转后）
        (0.0,   0.0,        0.11,  0.0),   # J3: 肘弯曲
        (0.0,   0.0,        0.10,  0.0),   # J4: 腕弯曲
        (0.0,  -np.pi/2,    0.04,  0.0),   # J5: 腕旋转
        (0.0,   np.pi/2,    0.035, 0.0),   # J6: 腕末端旋转
    ]

    # 从臂基座开始
    T = np.eye(4)
    T[:3, 3] = arm_base_offset  # 臂基座偏移

    for i, (a, alpha, d, theta_offset) in enumerate(dh_params):
        theta = joint_angles[i] + theta_offset
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)

        # 标准 DH 变换矩阵
        # T_i = [cosθ  -sinθ·cosα   sinθ·sinα   a·cosθ]
        #       [sinθ   cosθ·cosα  -cosθ·sinα   a·sinθ]
        #       [0      sinα        cosα         d      ]
        #       [0      0           0            1      ]
        Ti = np.array([
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [0,   sa,       ca,      d     ],
            [0,   0,        0,       1     ],
        ])
        T = T @ Ti

    # 末端执行器偏移（从 J6 到 ee_link，沿 Z 轴 0.04m）
    T_ee_offset = np.eye(4)
    T_ee_offset[2, 3] = 0.04
    T = T @ T_ee_offset

    return T


# ============================================================
# 第四部分：相机外参（安装偏移）
# ============================================================

# ---- 手腕相机安装偏移 ----
# 相机安装在末端执行器（J6 连杆）上，相对于末端执行器坐标系的偏移
# pos=(0.0, -0.065, 0.03): 相机在末端下方 6.5cm、前方 3cm
# rot=euler(-15°, 0, 0): 相机向下倾斜 15°
WRIST_CAMERA_OFFSET = euler_to_transform_matrix(
    roll=np.radians(-15.0),    # 向下倾斜 15°
    pitch=0.0,
    yaw=0.0,
    translation=np.array([0.0, -0.065, 0.03]),
)

# ---- 俯视相机安装偏移 ----
# 相机安装在 base_link 上，相对于 base_link 的固定偏移
# pos=(0.53, -0.55, 1.0): 在机器人前方 53cm、右侧 55cm、上方 1.0m
# rot=euler(-148°, 0, 0): 向下倾斜 148°（即从上方俯视，与水平面约成 58°）
TOP_CAMERA_OFFSET = euler_to_transform_matrix(
    roll=np.radians(-148.0),   # 俯视角度
    pitch=0.0,
    yaw=0.0,
    translation=np.array([0.53, -0.55, 1.0]),
)


# ============================================================
# 第五部分：核心转换类
# ============================================================

class CameraToBaseTransformer:
    """
    相机坐标 → 机器人基座坐标 转换器。

    使用流程：
      1. 初始化：创建转换器实例，设置相机内参
      2. 对于手腕相机：调用 transform_wrist_camera()，需要传入当前关节角度
      3. 对于俯视相机：调用 transform_top_camera()，不需要关节角度

    示例：
        >>> transformer = CameraToBaseTransformer()
        >>> # 俯视相机检测到目标在像素 (320, 240)，深度 0.5m
        >>> p_base = transformer.transform_top_camera(u=320, v=240, depth=0.5)
        >>> print(f"基座坐标: {p_base}")
        >>>
        >>> # 左腕相机检测到目标，需要当前关节角度
        >>> joints = [0.1, -0.5, 0.3, -0.2, 0.0, 0.1]
        >>> p_base = transformer.transform_wrist_camera(u=320, v=240, depth=0.3,
        ...                                              joint_angles=joints, arm="left")
    """

    def __init__(self, focal_length: float = 26.8, horizontal_aperture: float = 36.83,
                 width: int = 640, height: int = 480):
        """
        参数：
            focal_length: 焦距（mm），默认 26.8（X-Trainer 相机配置）
            horizontal_aperture: 水平光圈（mm），默认 36.83
            width: 图像宽度（像素），默认 640
            height: 图像高度（像素），默认 480
        """
        self.intrinsics = CameraIntrinsics(focal_length, horizontal_aperture, width, height)

    def transform_wrist_camera(self, u: float, v: float, depth: float,
                                joint_angles: np.ndarray, arm: str = "left") -> np.ndarray:
        """
        手腕相机：像素坐标 → 基座坐标。

        转换链：
          P_base = T_base_ee @ T_ee_camera @ P_cam

        其中：
          T_base_ee: 基座到末端执行器的变换（由正运动学计算，取决于关节角度）
          T_ee_camera: 末端执行器到相机的变换（固定的安装偏移）
          P_cam: 相机坐标系下的 3D 坐标（由像素 + 深度反投影得到）

        参数：
            u, v: 像素坐标
            depth: 深度值（米）
            joint_angles: 6 个关节角度（弧度）
            arm: "left" 或 "right"

        返回：
            [x, y, z] 基座坐标系下的 3D 坐标（米）
        """
        # 第 1 步：像素 → 相机坐标系 3D 坐标
        p_cam = self.intrinsics.pixel_to_camera(u, v, depth)

        # 第 2 步：相机坐标系 → 齐次坐标（补 1，变成 4x1 向量）
        p_cam_h = np.append(p_cam, 1.0)  # [x, y, z, 1]

        # 第 3 步：计算 T_base_ee（基座到末端执行器的变换）
        T_base_ee = compute_arm_fk(joint_angles, arm)

        # 第 4 步：计算完整变换 T_base_camera = T_base_ee @ T_ee_camera
        T_base_camera = T_base_ee @ WRIST_CAMERA_OFFSET

        # 第 5 步：变换到基座坐标系
        p_base_h = T_base_camera @ p_cam_h

        # 返回 3D 坐标（去掉齐次坐标的 1）
        return p_base_h[:3]

    def transform_top_camera(self, u: float, v: float, depth: float) -> np.ndarray:
        """
        俯视相机：像素坐标 → 基座坐标。

        俯视相机安装在 base_link 上，变换矩阵是固定的，不需要关节角度。

        转换链：
          P_base = T_base_camera @ P_cam

        其中 T_base_camera 是固定的（由俯视相机安装位置和朝向决定）。

        参数：
            u, v: 像素坐标
            depth: 深度值（米）

        返回：
            [x, y, z] 基座坐标系下的 3D 坐标（米）
        """
        # 第 1 步：像素 → 相机坐标系 3D 坐标
        p_cam = self.intrinsics.pixel_to_camera(u, v, depth)

        # 第 2 步：齐次坐标
        p_cam_h = np.append(p_cam, 1.0)

        # 第 3 步：用固定的俯视相机变换矩阵
        p_base_h = TOP_CAMERA_OFFSET @ p_cam_h

        return p_base_h[:3]

    def transform(self, u: float, v: float, depth: float,
                  camera: str, joint_angles: Optional[np.ndarray] = None) -> np.ndarray:
        """
        统一接口：根据相机名称自动选择转换方法。

        参数：
            u, v: 像素坐标
            depth: 深度值（米）
            camera: 相机名称，"left_wrist" / "right_wrist" / "top"
            joint_angles: 关节角度（手腕相机必须提供，俯视相机可省略）

        返回：
            [x, y, z] 基座坐标系下的 3D 坐标（米）
        """
        if camera == "top":
            return self.transform_top_camera(u, v, depth)
        elif camera in ("left_wrist", "right_wrist"):
            if joint_angles is None:
                raise ValueError(f"{camera} 需要提供 joint_angles 参数")
            arm = "left" if "left" in camera else "right"
            return self.transform_wrist_camera(u, v, depth, joint_angles, arm)
        else:
            raise ValueError(f"未知相机: {camera}，支持: top, left_wrist, right_wrist")

    def batch_transform(self, detections: list, camera: str,
                        joint_angles: Optional[np.ndarray] = None) -> np.ndarray:
        """
        批量转换：将多个检测结果一次性转换到基座坐标系。

        参数：
            detections: [(u, v, depth), ...] 检测结果列表
            camera: 相机名称
            joint_angles: 关节角度（手腕相机必须提供）

        返回：
            (N, 3) 数组，每行是一个目标的基座坐标
        """
        results = []
        for u, v, depth in detections:
            p = self.transform(u, v, depth, camera, joint_angles)
            results.append(p)
        return np.array(results)

    def get_camera_pose_in_base(self, camera: str,
                                 joint_angles: Optional[np.ndarray] = None) -> np.ndarray:
        """
        获取相机在基座坐标系中的位姿（4x4 变换矩阵）。
        用于可视化或调试。

        参数：
            camera: 相机名称
            joint_angles: 关节角度（手腕相机必须提供）

        返回：
            4x4 齐次变换矩阵
        """
        if camera == "top":
            return TOP_CAMERA_OFFSET.copy()
        elif camera in ("left_wrist", "right_wrist"):
            if joint_angles is None:
                raise ValueError(f"{camera} 需要提供 joint_angles")
            arm = "left" if "left" in camera else "right"
            T_base_ee = compute_arm_fk(joint_angles, arm)
            return T_base_ee @ WRIST_CAMERA_OFFSET
        else:
            raise ValueError(f"未知相机: {camera}")


# ============================================================
# 第六部分：测试函数
# ============================================================

def test_coordinate_transform():
    """
    测试坐标转换模块的正确性。

    测试内容：
      1. 相机内参反投影/正投影一致性
      2. 俯视相机变换（固定变换）
      3. 手腕相机变换（依赖关节角度）
      4. 旋转矩阵正交性验证
    """
    print("=" * 60)
    print("坐标转换模块测试")
    print("=" * 60)

    transformer = CameraToBaseTransformer()
    all_pass = True

    # ---- 测试 1：内参反投影一致性 ----
    print("\n[测试 1] 相机内参：像素 → 3D → 像素 一致性")
    intr = transformer.intrinsics
    print(f"  内参: {intr}")

    # 在像素 (200, 150) 处，深度 0.5m 的点
    u_orig, v_orig, depth = 200.0, 150.0, 0.5
    p_cam = intr.pixel_to_camera(u_orig, v_orig, depth)
    u_back, v_back = intr.camera_to_pixel(p_cam)

    pixel_error = abs(u_orig - u_back) + abs(v_orig - v_back)
    passed = pixel_error < 0.01
    all_pass = all_pass and passed
    print(f"  原始像素: ({u_orig}, {v_orig})")
    print(f"  相机坐标: ({p_cam[0]:.4f}, {p_cam[1]:.4f}, {p_cam[2]:.4f})")
    print(f"  反投影像素: ({u_back:.2f}, {v_back:.2f})")
    print(f"  误差: {pixel_error:.6f} 像素")
    print(f"  结果: {'PASS' if passed else 'FAIL'}")

    # ---- 测试 2：俯视相机变换 ----
    print("\n[测试 2] 俯视相机：像素 (320, 240) + 深度 1.0m → 基座坐标")
    p_base = transformer.transform_top_camera(320, 240, 1.0)
    # 俯视相机在 base_link 上方 1m，像素中心应该对应相机正下方的点
    print(f"  基座坐标: ({p_base[0]:.4f}, {p_base[1]:.4f}, {p_base[2]:.4f})")
    passed = np.all(np.isfinite(p_base))
    all_pass = all_pass and passed
    print(f"  有限性检查: {'PASS' if passed else 'FAIL'}")

    # ---- 测试 3：手腕相机变换 ----
    print("\n[测试 3] 左腕相机：像素 (320, 240) + 深度 0.3m + 零关节 → 基座坐标")
    joints_zero = np.zeros(6)
    p_base_wrist = transformer.transform_wrist_camera(
        320, 240, 0.3, joint_angles=joints_zero, arm="left"
    )
    print(f"  基座坐标: ({p_base_wrist[0]:.4f}, {p_base_wrist[1]:.4f}, {p_base_wrist[2]:.4f})")
    passed = np.all(np.isfinite(p_base_wrist))
    all_pass = all_pass and passed
    print(f"  有限性检查: {'PASS' if passed else 'FAIL'}")

    # ---- 测试 4：不同关节角度下同一像素应映射到不同基座坐标 ----
    print("\n[测试 4] 关节角度影响验证")
    joints_a = np.array([0.0, -0.5, 0.3, 0.0, 0.0, 0.0])
    joints_b = np.array([0.5, -0.3, 0.5, 0.0, 0.0, 0.0])
    p_a = transformer.transform_wrist_camera(320, 240, 0.3, joints_a, "left")
    p_b = transformer.transform_wrist_camera(320, 240, 0.3, joints_b, "left")
    diff = np.linalg.norm(p_a - p_b)
    passed = diff > 0.001  # 不同关节角度应该得到不同结果
    all_pass = all_pass and passed
    print(f"  关节 A 基座坐标: ({p_a[0]:.4f}, {p_a[1]:.4f}, {p_a[2]:.4f})")
    print(f"  关节 B 基座坐标: ({p_b[0]:.4f}, {p_b[1]:.4f}, {p_b[2]:.4f})")
    print(f"  差异: {diff:.4f}m")
    print(f"  结果: {'PASS' if passed else 'FAIL'}")

    # ---- 测试 5：旋转矩阵正交性 ----
    print("\n[测试 5] 旋转矩阵正交性")
    R = euler_to_rotation_matrix(0.3, -0.5, 0.7)
    should_be_I = R @ R.T
    ortho_error = np.linalg.norm(should_be_I - np.eye(3))
    passed = ortho_error < 1e-10
    all_pass = all_pass and passed
    print(f"  R @ R.T 与单位矩阵误差: {ortho_error:.2e}")
    print(f"  结果: {'PASS' if passed else 'FAIL'}")

    # ---- 测试 6：统一接口 ----
    print("\n[测试 6] 统一接口 transform()")
    p1 = transformer.transform(320, 240, 0.5, "top")
    p2 = transformer.transform(320, 240, 0.3, "left_wrist", np.zeros(6))
    p3 = transformer.transform(320, 240, 0.3, "right_wrist", np.zeros(6))
    passed = all(np.all(np.isfinite(p)) for p in [p1, p2, p3])
    all_pass = all_pass and passed
    print(f"  top:        ({p1[0]:.4f}, {p1[1]:.4f}, {p1[2]:.4f})")
    print(f"  left_wrist: ({p2[0]:.4f}, {p2[1]:.4f}, {p2[2]:.4f})")
    print(f"  right_wrist:({p3[0]:.4f}, {p3[1]:.4f}, {p3[2]:.4f})")
    print(f"  结果: {'PASS' if passed else 'FAIL'}")

    # ---- 测试 7：批量转换 ----
    print("\n[测试 7] 批量转换 batch_transform()")
    detections = [(100, 100, 0.3), (320, 240, 0.5), (500, 400, 0.4)]
    batch_result = transformer.batch_transform(detections, "top")
    passed = batch_result.shape == (3, 3) and np.all(np.isfinite(batch_result))
    all_pass = all_pass and passed
    print(f"  输入: {len(detections)} 个检测")
    print(f"  输出形状: {batch_result.shape}")
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
    test_coordinate_transform()
