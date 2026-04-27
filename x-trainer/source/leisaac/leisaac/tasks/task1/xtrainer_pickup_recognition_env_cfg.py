import torch
from pathlib import Path

from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.sensors import FrameTransformerCfg
import isaaclab.sim as sim_utils

from leisaac.utils.constant import ASSETS_ROOT
from leisaac.utils.general_assets import parse_usd_and_create_subassets
from leisaac.utils.domain_randomization import domain_randomization

from . import mdp
from ..template import XTrainerArmTaskSceneCfg, XTrainerArmTaskEnvCfg, XTrainerArmTerminationsCfg, XTrainerArmObservationsCfg

# ==============================================================================
# Task1: XTrainer 抓取识别环境配置
# ==============================================================================
# 说明：
# - 该文件负责场景加载、可重置资产注册、以及 reset 时的物体随机化。
# - 物体初始位姿由 parse_usd_and_create_subassets 注册到 scene，按 reset_all 统一重置。

TRAINING_ENV_USD_PATH = str(Path(ASSETS_ROOT) / "scenes" / "task1" / "training_env.usd")

TABLE_WITH_GOODS_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=TRAINING_ENV_USD_PATH,
    )
)

# ------------------------------------------------------------------------------
# Task1 可视化默认参数（用于 --enable_visualization）
# 你后续主要在这里调整判定区域和判定点补偿
# ------------------------------------------------------------------------------
TASK1_VISUALIZATION_PLATE_NAME = "bearingset"
TASK1_VISUALIZATION_OBJECT_NAMES = (
    "factory_nut_loose",
    "factory_nut_loose_01",
    "factory_nut_loose_02",
)
TASK1_VISUALIZATION_X_RANGE = (-0.15, 0.15)
TASK1_VISUALIZATION_Y_RANGE = (-0.005, 0.015)
TASK1_VISUALIZATION_HEIGHT_RANGE = (-0.025, 0.025)
TASK1_VISUALIZATION_OBJECT_OFFSETS = {
    "factory_nut_loose": (0, 0, 0.05),
    "factory_nut_loose_01": (0, 0, 0.05),
    "factory_nut_loose_02": (0, 0, 0.05),
}
# 三个独立判定区域（相对 bearingset 的偏移范围）
# 三个螺母任意一一占满三个区域即可，不要求物体名和区域名对应
TASK1_VISUALIZATION_OBJECT_REGIONS = {
    "zone_1": {
        "x_range": (-0.18, -0.2),
        "y_range": (-0.005, 0.01),
        "height_range": (-0.025, 0.025),
    },
    "zone_2": {
        "x_range": (-0.01, 0.01),
        "y_range": (-0.005, 0.01),
        "height_range": (-0.025, 0.025),
    },
    "zone_3": {
        "x_range": (0.19, 0.21),
        "y_range": (-0.005, 0.01),
        "height_range": (-0.025, 0.025),
    },
}
TASK1_OFFSET_FRAME = "world"  # "world" 或 "local"

TASK1_SUCCESS_PLATE_NAME = TASK1_VISUALIZATION_PLATE_NAME
TASK1_SUCCESS_OBJECT_NAMES = TASK1_VISUALIZATION_OBJECT_NAMES
TASK1_SUCCESS_X_RANGE = TASK1_VISUALIZATION_X_RANGE
TASK1_SUCCESS_Y_RANGE = TASK1_VISUALIZATION_Y_RANGE
TASK1_SUCCESS_HEIGHT_RANGE = TASK1_VISUALIZATION_HEIGHT_RANGE
TASK1_SUCCESS_OBJECT_OFFSETS = TASK1_VISUALIZATION_OBJECT_OFFSETS
TASK1_SUCCESS_OBJECT_REGIONS = TASK1_VISUALIZATION_OBJECT_REGIONS
TASK1_SUCCESS_OFFSET_FRAME = TASK1_OFFSET_FRAME


@configclass
class PickupRecognitionSceneCfg(XTrainerArmTaskSceneCfg):
    """Task1 场景定义。"""

    # 主场景 USD
    scene: AssetBaseCfg = TABLE_WITH_GOODS_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

    # 左臂末端参考系
    left_ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/x_trainer_asm_0226_SLDASM/J1_6",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/x_trainer_asm_0226_SLDASM/J1_6",
                name="left_flange"
            ),
        ],
    )

    # 右臂末端参考系
    right_ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/x_trainer_asm_0226_SLDASM/J2_6",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/x_trainer_asm_0226_SLDASM/J2_6",
                name="right_grasp_center",
            ),
        ],
    )


@configclass
class Task1TerminationsCfg(XTrainerArmTerminationsCfg):
    """Task1 终止配置。"""

    success = DoneTerm(
        func=mdp.task_done,
        params={
            "objects_cfg": [SceneEntityCfg(name) for name in TASK1_SUCCESS_OBJECT_NAMES],
            "plate_cfg": SceneEntityCfg(TASK1_SUCCESS_PLATE_NAME),
            "x_range": TASK1_SUCCESS_X_RANGE,
            "y_range": TASK1_SUCCESS_Y_RANGE,
            "height_range": TASK1_SUCCESS_HEIGHT_RANGE,
            "object_offsets": TASK1_SUCCESS_OBJECT_OFFSETS,
            "object_regions": TASK1_SUCCESS_OBJECT_REGIONS,
            "offset_frame": TASK1_SUCCESS_OFFSET_FRAME,
            "verbose": True,
            "visualize": True,
        },
    )


@configclass
class Task1EnvCfg(XTrainerArmTaskEnvCfg):
    """Task1 总环境配置。"""

    scene: PickupRecognitionSceneCfg = PickupRecognitionSceneCfg(env_spacing=8.0)
    observations: XTrainerArmObservationsCfg = XTrainerArmObservationsCfg()
    terminations: Task1TerminationsCfg = Task1TerminationsCfg()
    enable_visualization: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()

        # 机器人初始位姿
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.1)
        self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)

        # 仿真步长
        self.sim.dt = 1.0 / 120.0
        self.sim.physx.enable_ccd = True

        # 注册需要纳入 reset 管理的刚体/关节体（名称精确匹配）
        reset_managed_parts = [
            "bearingset",
            "factory_nut_loose",
            "factory_nut_loose_01",
            "factory_nut_loose_02",
        ]
        parse_usd_and_create_subassets(
            TRAINING_ENV_USD_PATH,
            self,
            specific_name_list=reset_managed_parts,
            pose_reference="scene_root",
        )

        # 每次 reset 需要做网格随机化的目标物体
        random_parts = [
            "factory_nut_loose",
            "factory_nut_loose_01",
            "factory_nut_loose_02",
        ]

        random_opts = []
        if random_parts:
            random_opts.append(
                EventTermCfg(
                    func=reset_tube_grid,
                    mode="reset",
                    params={
                        "asset_names": random_parts,
                        "x_range": (-0.1, 0.4),
                        "y_range": (-0.15, 0.05),
                        "grid_shape": (1, 3),
                        "min_center_distance": 0.055,
                        "max_sample_attempts": 24,
                    },
                )
            )

        domain_randomization(self, random_options=random_opts)


def reset_tube_grid(
    env,
    env_ids: torch.Tensor,
    asset_names: list[str],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    grid_shape: tuple[int, int] = (1, 3),
    min_center_distance: float = 0.05,
    max_sample_attempts: int = 20,
):
    """
    将多个物体随机分配到网格中，尽量避免重叠。

    Args:
        env: 环境对象
        env_ids: 本次需要 reset 的环境索引
        asset_names: 需要随机化的资产名称列表
        x_range: 相对默认位姿的 x 偏移范围
        y_range: 相对默认位姿的 y 偏移范围
        grid_shape: 网格形状 (rows, cols)
        min_center_distance: 物体中心最小间距（米）
        max_sample_attempts: 每个环境最大重采样次数
    """
    if not asset_names:
        return

    num_envs = env.num_envs
    if env_ids is None:
        env_ids = torch.arange(num_envs, device=env.device)

    num_reset = len(env_ids)
    rows, cols = grid_shape
    num_cells = rows * cols

    if len(asset_names) > num_cells:
        raise ValueError(f"Asset count ({len(asset_names)}) exceeds grid cells ({num_cells})!")

    x_min, x_max = x_range
    y_min, y_max = y_range

    # 修正：网格宽高按 x/cols, y/rows 计算，避免落点轴向错位
    cell_width = (x_max - x_min) / cols
    cell_height = (y_max - y_min) / rows

    x_indices = torch.arange(cols, device=env.device).repeat(rows)
    y_indices = torch.arange(rows, device=env.device).repeat_interleave(cols)

    grid_centers_x = x_min + (x_indices + 0.5) * cell_width
    grid_centers_y = y_min + (y_indices + 0.5) * cell_height
    grid_centers = torch.stack([grid_centers_x, grid_centers_y], dim=-1)

    rand_noise = torch.rand((num_reset, num_cells), device=env.device)
    cell_indices = torch.argsort(rand_noise, dim=-1)[:, :len(asset_names)]

    # 每个环境对应每个物体的网格中心，shape: (num_reset, num_assets, 2)
    assigned_centers = grid_centers[cell_indices]

    # 在网格内抖动，并通过重采样保证最小中心间距
    max_offset_x = cell_width * 0.20
    max_offset_y = cell_height * 0.20
    num_assets = len(asset_names)
    sampled_offsets = assigned_centers.clone()
    offset_scale = torch.tensor([max_offset_x, max_offset_y], device=env.device)

    # 最小间距不能大于相邻格中心最小距离，否则会无解
    effective_min_distance = min(min_center_distance, min(cell_width, cell_height) * 0.9)

    for env_idx in range(num_reset):
        centers = assigned_centers[env_idx]
        if num_assets == 1:
            noise = 2 * torch.rand((1, 2), device=env.device) - 1
            sampled_offsets[env_idx] = centers + noise * offset_scale
            continue

        success = False
        for _ in range(max_sample_attempts):
            noise = 2 * torch.rand((num_assets, 2), device=env.device) - 1
            candidate = centers + noise * offset_scale
            pairwise = torch.cdist(candidate.unsqueeze(0), candidate.unsqueeze(0)).squeeze(0)
            pairwise.fill_diagonal_(float("inf"))
            if torch.all(pairwise >= effective_min_distance):
                sampled_offsets[env_idx] = candidate
                success = True
                break

        # 回退策略：若多次重采样失败，使用网格中心（严格不重叠）
        if not success:
            sampled_offsets[env_idx] = centers

    for i, asset_name in enumerate(asset_names):
        asset = env.scene[asset_name]
        root_state = asset.data.default_root_state[env_ids].clone()
        root_state[:, 0] += sampled_offsets[:, i, 0]
        root_state[:, 1] += sampled_offsets[:, i, 1]
        root_state[:, 7:] = 0.0
        asset.write_root_state_to_sim(root_state, env_ids=env_ids)
