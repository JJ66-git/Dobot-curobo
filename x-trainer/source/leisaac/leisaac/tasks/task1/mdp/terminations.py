from __future__ import annotations

from typing import List, Optional
from itertools import permutations

import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv, DirectRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

from leisaac.utils.robot_utils import is_xtrainer_at_rest_pose
from .observations import get_object_pose_w, init_task1_visualization, update_task1_visualization


def task_done(
    env: ManagerBasedRLEnv | DirectRLEnv,
    objects_cfg: List[SceneEntityCfg],
    plate_cfg: SceneEntityCfg,
    x_range: tuple[float, float] = (-0.10, 0.10),
    y_range: tuple[float, float] = (-0.10, 0.10),
    height_range: tuple[float, float] = (-0.05, 0.05),
    object_offsets: Optional[dict[str, tuple[float, float, float]]] = None,
    object_regions: Optional[dict[str, dict[str, tuple[float, float]]]] = None,
    offset_frame: str = "local",
    verbose: bool = False,
    visualize: bool = False,
) -> torch.Tensor:
    """Task1 成功判定：目标物体任意一一占满所有判定区域，且机械臂回到 rest pose。"""
    offsets = object_offsets or {}
    use_offset_frame = "local" if offset_frame == "local" else "world"

    if visualize and getattr(env.cfg, "enable_visualization", False):
        init_task1_visualization(
            env=env,
            plate_name=plate_cfg.name,
            object_names=tuple(cfg.name for cfg in objects_cfg),
            x_range=x_range,
            y_range=y_range,
            height_range=height_range,
            object_offsets=offsets,
            object_regions=object_regions,
            offset_frame=use_offset_frame,
        )
        update_task1_visualization(env)

    done = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    plate: RigidObject = env.scene[plate_cfg.name]
    plate_pos_w, _ = get_object_pose_w(plate, env_id=None, prefer_link_frame=True)
    plate_x = plate_pos_w[:, 0] - env.scene.env_origins[:, 0]
    plate_y = plate_pos_w[:, 1] - env.scene.env_origins[:, 1]
    plate_z = plate_pos_w[:, 2] - env.scene.env_origins[:, 2]

    object_positions: list[torch.Tensor] = []
    for object_cfg in objects_cfg:
        object_entity: RigidObject = env.scene[object_cfg.name]
        object_pos_w, object_quat_w = get_object_pose_w(object_entity, env_id=None, prefer_link_frame=True)
        object_pos_w = object_pos_w.clone()

        object_offset = offsets.get(object_cfg.name, (0.0, 0.0, 0.0))
        if object_offset != (0.0, 0.0, 0.0):
            offset_tensor = torch.tensor(object_offset, device=env.device, dtype=object_pos_w.dtype).unsqueeze(0)
            offset_tensor = offset_tensor.repeat(env.num_envs, 1)
            if use_offset_frame == "local":
                offset_world = quat_apply(object_quat_w, offset_tensor)
            else:
                offset_world = offset_tensor
            object_pos_w = object_pos_w + offset_world

        object_positions.append(object_pos_w)

    if object_regions:
        region_defs = list(object_regions.values())
        num_objects = len(object_positions)
        num_regions = len(region_defs)
        if num_objects != num_regions:
            done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        else:
            in_region_matrix: list[list[torch.Tensor]] = []
            for object_pos_w in object_positions:
                object_x = object_pos_w[:, 0] - env.scene.env_origins[:, 0]
                object_y = object_pos_w[:, 1] - env.scene.env_origins[:, 1]
                object_z = object_pos_w[:, 2] - env.scene.env_origins[:, 2]

                row: list[torch.Tensor] = []
                for region_cfg in region_defs:
                    region_x = region_cfg.get("x_range", x_range)
                    region_y = region_cfg.get("y_range", y_range)
                    region_h = region_cfg.get("height_range", height_range)

                    in_x = torch.logical_and(object_x > plate_x + region_x[0], object_x < plate_x + region_x[1])
                    in_y = torch.logical_and(object_y > plate_y + region_y[0], object_y < plate_y + region_y[1])
                    in_z = torch.logical_and(object_z > plate_z + region_h[0], object_z < plate_z + region_h[1])
                    row.append(torch.logical_and(torch.logical_and(in_x, in_y), in_z))
                in_region_matrix.append(row)

            match_ok = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            for perm in permutations(range(num_regions), num_objects):
                assign_ok = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
                for object_idx, region_idx in enumerate(perm):
                    assign_ok = torch.logical_and(assign_ok, in_region_matrix[object_idx][region_idx])
                match_ok = torch.logical_or(match_ok, assign_ok)
            done = torch.logical_and(done, match_ok)
    else:
        # 兼容旧逻辑：所有物体都在同一个区域即可
        for object_pos_w in object_positions:
            object_x = object_pos_w[:, 0] - env.scene.env_origins[:, 0]
            object_y = object_pos_w[:, 1] - env.scene.env_origins[:, 1]
            object_z = object_pos_w[:, 2] - env.scene.env_origins[:, 2]

            done = torch.logical_and(done, object_x > plate_x + x_range[0])
            done = torch.logical_and(done, object_x < plate_x + x_range[1])
            done = torch.logical_and(done, object_y > plate_y + y_range[0])
            done = torch.logical_and(done, object_y < plate_y + y_range[1])
            done = torch.logical_and(done, object_z > plate_z + height_range[0])
            done = torch.logical_and(done, object_z < plate_z + height_range[1])

    joint_pos = env.scene["robot"].data.joint_pos
    joint_names = env.scene["robot"].data.joint_names
    done = torch.logical_and(done, is_xtrainer_at_rest_pose(joint_pos, joint_names))

    if verbose:
        for env_id in range(env.num_envs):
            if done[env_id]:
                print(f"[Env {env_id}] ✓ task1 success")

    return done
