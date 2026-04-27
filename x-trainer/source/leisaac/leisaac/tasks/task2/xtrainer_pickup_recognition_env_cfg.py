from pathlib import Path

from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

from leisaac.utils.constant import ASSETS_ROOT
from leisaac.utils.general_assets import parse_usd_and_create_subassets

from . import mdp
from ..template import XTrainerArmTaskEnvCfg, XTrainerArmTaskSceneCfg, XTrainerArmTerminationsCfg, XTrainerArmObservationsCfg


TRAINING_ENV_USD_PATH = str(Path(ASSETS_ROOT) / "scenes" / "task2" / "training_env.usd")

TABLE_WITH_GOODS_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=TRAINING_ENV_USD_PATH,
    )
)

# ------------------------------------------------------------------------------
# Task2 可视化与成功判定参数
# - 物体不随机化，仅支持 reset 到默认状态
# - 三类物体对应三个判定区域（均相对 teatable）
# ------------------------------------------------------------------------------
TASK2_VISUALIZATION_CONTAINER_NAME = "teatable"
TASK2_VISUALIZATION_OBJECT_NAMES = (
    "cup_1",
    "cup_2",
    "cup_3",
    "cup_4",
    "bigcup",
    "teaholder",
)

TASK2_VISUALIZATION_X_RANGE = (-0.10, 0.10)
TASK2_VISUALIZATION_Y_RANGE = (-0.10, 0.10)
TASK2_VISUALIZATION_HEIGHT_RANGE = (-0.05, 0.05)
TASK2_VISUALIZATION_OBJECT_OFFSETS = {name: (0.0, 0.0, 0.0) for name in TASK2_VISUALIZATION_OBJECT_NAMES}

TASK2_VISUALIZATION_OBJECT_REGIONS = {
    "cups_zone": {
        "x_range": (-0.225, -0.05),
        "y_range": (0.005, 0.18),
        "height_range": (-0.02, 0.08),
    },
    "bigcup_zone": {
        "x_range": (-0.33, -0.25),
        "y_range": (0, 0.12),
        "height_range": (-0.02, 0.10),
    },
    "teaholder_zone": {
        "x_range": (0.06, 0.13),
        "y_range": (0.09, 0.16),
        "height_range": (-0.02, 0.10),
    },
}

TASK2_VISUALIZATION_OBJECT_REGION_MAP = {
    "cup_1": "cups_zone",
    "cup_2": "cups_zone",
    "cup_3": "cups_zone",
    "cup_4": "cups_zone",
    "bigcup": "bigcup_zone",
    "teaholder": "teaholder_zone",
}

TASK2_OFFSET_FRAME = "world"  # "world" 或 "local"

TASK2_SUCCESS_CONTAINER_NAME = TASK2_VISUALIZATION_CONTAINER_NAME
TASK2_SUCCESS_OBJECT_NAMES = TASK2_VISUALIZATION_OBJECT_NAMES
TASK2_SUCCESS_X_RANGE = TASK2_VISUALIZATION_X_RANGE
TASK2_SUCCESS_Y_RANGE = TASK2_VISUALIZATION_Y_RANGE
TASK2_SUCCESS_HEIGHT_RANGE = TASK2_VISUALIZATION_HEIGHT_RANGE
TASK2_SUCCESS_OBJECT_OFFSETS = TASK2_VISUALIZATION_OBJECT_OFFSETS
TASK2_SUCCESS_OBJECT_REGIONS = TASK2_VISUALIZATION_OBJECT_REGIONS
TASK2_SUCCESS_OBJECT_REGION_MAP = TASK2_VISUALIZATION_OBJECT_REGION_MAP
TASK2_SUCCESS_OFFSET_FRAME = TASK2_OFFSET_FRAME


@configclass
class PickupRecognitionSceneCfg(XTrainerArmTaskSceneCfg):
    """Task2 场景定义。"""

    scene: AssetBaseCfg = TABLE_WITH_GOODS_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

    left_ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/x_trainer_asm_0226_SLDASM/J1_6",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/x_trainer_asm_0226_SLDASM/J1_6",
                name="left_flange",
            ),
        ],
    )

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
class Task2TerminationsCfg(XTrainerArmTerminationsCfg):
    """Task2 终止配置。"""

    success = DoneTerm(
        func=mdp.task_done,
        params={
            "object_names": TASK2_SUCCESS_OBJECT_NAMES,
            "container_name": TASK2_SUCCESS_CONTAINER_NAME,
            "x_range": TASK2_SUCCESS_X_RANGE,
            "y_range": TASK2_SUCCESS_Y_RANGE,
            "height_range": TASK2_SUCCESS_HEIGHT_RANGE,
            "object_offsets": TASK2_SUCCESS_OBJECT_OFFSETS,
            "object_regions": TASK2_SUCCESS_OBJECT_REGIONS,
            "object_region_map": TASK2_SUCCESS_OBJECT_REGION_MAP,
            "offset_frame": TASK2_SUCCESS_OFFSET_FRAME,
            "verbose": True,
            "visualize": True,
        },
    )


@configclass
class Task2EnvCfg(XTrainerArmTaskEnvCfg):
    """Task2 总环境配置。"""

    scene: PickupRecognitionSceneCfg = PickupRecognitionSceneCfg(env_spacing=8.0)
    observations: XTrainerArmObservationsCfg = XTrainerArmObservationsCfg()
    terminations: Task2TerminationsCfg = Task2TerminationsCfg()
    enable_visualization: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.robot.init_state.pos = (0.0, 0.0, 0.1)
        self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)

        self.sim.dt = 1.0 / 120.0
        self.sim.physx.enable_ccd = True

        # 不做随机化，仅注册可 reset 的资产
        reset_managed_parts = [
            TASK2_VISUALIZATION_CONTAINER_NAME,
            *TASK2_VISUALIZATION_OBJECT_NAMES,
        ]
        parse_usd_and_create_subassets(
            TRAINING_ENV_USD_PATH,
            self,
            specific_name_list=reset_managed_parts,
            pose_reference="scene_root",
        )
