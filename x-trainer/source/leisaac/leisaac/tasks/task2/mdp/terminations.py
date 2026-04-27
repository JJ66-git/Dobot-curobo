from __future__ import annotations

from typing import List, Optional

import torch

from isaaclab.envs import ManagerBasedRLEnv, DirectRLEnv

from leisaac.utils.robot_utils import is_xtrainer_at_rest_pose
from .observations import (
    _get_scene_entity_pos_w,
    get_detection_point_pos_w,
    init_task2_visualization,
    update_task2_visualization,
)


def task_done(
    env: ManagerBasedRLEnv | DirectRLEnv,
    object_names: List[str],
    container_name: str,
    x_range: tuple[float, float] = (-0.10, 0.10),
    y_range: tuple[float, float] = (-0.10, 0.10),
    height_range: tuple[float, float] = (-0.05, 0.05),
    object_offsets: Optional[dict[str, tuple[float, float, float]]] = None,
    object_regions: Optional[dict[str, dict[str, tuple[float, float]]]] = None,
    object_region_map: Optional[dict[str, str]] = None,
    offset_frame: str = "local",
    verbose: bool = False,
    visualize: bool = False,
) -> torch.Tensor:
    """Task2 成功判定：两类物体分别占据对应判定区域，且机械臂回到 rest pose。"""
    offsets = object_offsets or {}
    regions = object_regions or {}
    region_map = object_region_map or {}
    use_offset_frame = "local" if offset_frame == "local" else "world"

    if visualize and getattr(env.cfg, "enable_visualization", False):
        init_task2_visualization(
            env=env,
            container_name=container_name,
            object_names=tuple(object_names),
            x_range=x_range,
            y_range=y_range,
            height_range=height_range,
            object_offsets=offsets,
            object_regions=regions,
            object_region_map=region_map,
            offset_frame=use_offset_frame,
            apply_object_scale=True,
        )
        update_task2_visualization(env)

    done = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    container_pos_w = _get_scene_entity_pos_w(env, container_name)
    container_x = container_pos_w[:, 0] - env.scene.env_origins[:, 0]
    container_y = container_pos_w[:, 1] - env.scene.env_origins[:, 1]
    container_z = container_pos_w[:, 2] - env.scene.env_origins[:, 2]

    region_success: dict[str, torch.Tensor] = {}
    for region_name in regions.keys():
        region_success[region_name] = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    region_has_candidate: dict[str, bool] = {region_name: False for region_name in regions.keys()}

    for object_name in object_names:
        object_entity = env.scene[object_name]
        object_offset = offsets.get(object_name, (0.0, 0.0, 0.0))
        object_pos_w = get_detection_point_pos_w(
            env=env,
            object_entity=object_entity,
            object_offset=object_offset,
            offset_frame=use_offset_frame,
            apply_object_scale=True,
            use_visual_center_anchor=True,
        )

        object_x = object_pos_w[:, 0] - env.scene.env_origins[:, 0]
        object_y = object_pos_w[:, 1] - env.scene.env_origins[:, 1]
        object_z = object_pos_w[:, 2] - env.scene.env_origins[:, 2]

        region_name = region_map.get(object_name)
        region_cfg = regions.get(region_name, {}) if region_name is not None else {}
        region_x = region_cfg.get("x_range", x_range)
        region_y = region_cfg.get("y_range", y_range)
        region_h = region_cfg.get("height_range", height_range)

        in_region = torch.logical_and(object_x > container_x + region_x[0], object_x < container_x + region_x[1])
        in_region = torch.logical_and(in_region, object_y > container_y + region_y[0])
        in_region = torch.logical_and(in_region, object_y < container_y + region_y[1])
        in_region = torch.logical_and(in_region, object_z > container_z + region_h[0])
        in_region = torch.logical_and(in_region, object_z < container_z + region_h[1])

        if region_name in region_success:
            region_success[region_name] = torch.logical_or(region_success[region_name], in_region)
            region_has_candidate[region_name] = True
        else:
            done = torch.logical_and(done, in_region)

    if region_success:
        for region_name, region_ok in region_success.items():
            if not region_has_candidate[region_name]:
                done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
                break
            done = torch.logical_and(done, region_ok)

    joint_pos = env.scene["robot"].data.joint_pos
    joint_names = env.scene["robot"].data.joint_names
    done = torch.logical_and(done, is_xtrainer_at_rest_pose(joint_pos, joint_names))

    if verbose:
        for env_id in range(env.num_envs):
            if done[env_id]:
                print(f"[Env {env_id}] ✓ task2 success")

    return done
