import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv, DirectRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

try:
    from pxr import UsdGeom, Gf, Usd, UsdShade, Sdf
    USD_AVAILABLE = True
except ImportError:
    USD_AVAILABLE = False
    UsdShade = None
    Sdf = None


def object_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    diff_threshold: float = 0.05,
    grasp_threshold: float = 0.01,
) -> torch.Tensor:
    """检查指定物体是否被抓取。"""
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    object_entity: RigidObject = env.scene[object_cfg.name]

    object_pos = object_entity.data.root_pos_w
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[:, 0, :]

    offset_local = torch.tensor([0.0, 0.0, 0.16], device=env.device).repeat(env.num_envs, 1)
    grasp_center_pos = ee_pos_w + quat_apply(ee_quat_w, offset_local)
    pos_diff = torch.linalg.vector_norm(object_pos - grasp_center_pos, dim=1)

    joint_ids, _ = robot.find_joints("J2_8")
    is_gripper_closed = robot.data.joint_pos[:, joint_ids[0]] > grasp_threshold
    return torch.logical_and(pos_diff < diff_threshold, is_gripper_closed)


def object_in_container(
    env: ManagerBasedRLEnv | DirectRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    container_cfg: SceneEntityCfg = SceneEntityCfg("container"),
    x_range: tuple[float, float] = (-0.10, 0.10),
    y_range: tuple[float, float] = (-0.10, 0.10),
    z_range: tuple[float, float] = (0.0, 0.20),
) -> torch.Tensor:
    """检查物体是否位于容器范围内。"""
    object_entity: RigidObject = env.scene[object_cfg.name]
    container: RigidObject = env.scene[container_cfg.name]

    rel_pos = object_entity.data.root_pos_w - container.data.root_pos_w
    in_x = torch.logical_and(rel_pos[:, 0] > x_range[0], rel_pos[:, 0] < x_range[1])
    in_y = torch.logical_and(rel_pos[:, 1] > y_range[0], rel_pos[:, 1] < y_range[1])
    in_z = torch.logical_and(rel_pos[:, 2] > z_range[0], rel_pos[:, 2] < z_range[1])
    return torch.logical_and(torch.logical_and(in_x, in_y), in_z)


def get_object_pose_w(
    object_entity: RigidObject,
    env_id: int | None = None,
    prefer_link_frame: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """获取对象世界位姿，优先 root_link（若可用），否则 root。"""
    data = object_entity.data
    has_link_pose = hasattr(data, "root_link_pos_w") and hasattr(data, "root_link_quat_w")

    if prefer_link_frame and has_link_pose:
        pos_all = data.root_link_pos_w
        quat_all = data.root_link_quat_w
    else:
        pos_all = data.root_pos_w
        quat_all = data.root_quat_w

    if env_id is None:
        return pos_all, quat_all
    return pos_all[env_id], quat_all[env_id]


def _get_env_base_path(env: ManagerBasedRLEnv | DirectRLEnv, entity_name: str) -> str:
    try:
        entity = env.scene[entity_name]
        prim_paths = entity.root_physx_view.prim_paths
        if prim_paths:
            parts = prim_paths[0].split("/")
            if len(parts) >= 4:
                return "/".join(parts[:4])
    except Exception:
        pass
    return "/World/envs/env_0"


def _compute_detection_point_world_pos(
    object_entity: RigidObject,
    env_id: int,
    object_offset: tuple[float, float, float],
    offset_frame: str = "local",
) -> tuple[float, float, float]:
    object_pos, object_quat = get_object_pose_w(object_entity, env_id=env_id, prefer_link_frame=True)
    object_pos = object_pos.cpu()
    offset_tensor = torch.tensor(object_offset, dtype=object_pos.dtype)

    if offset_frame == "local":
        offset_world = quat_apply(object_quat.cpu().unsqueeze(0), offset_tensor.unsqueeze(0))[0]
    else:
        offset_world = offset_tensor

    point = object_pos + offset_world
    return (float(point[0]), float(point[1]), float(point[2]))


def _set_or_create_translate_op_from_world(prim, world_pos: tuple[float, float, float]) -> None:
    local_pos = Gf.Vec3d(*world_pos)
    try:
        parent_prim = prim.GetParent()
        if parent_prim and parent_prim.IsValid():
            parent_world = UsdGeom.Xformable(parent_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            local_pos = parent_world.GetInverse().Transform(local_pos)
    except Exception:
        local_pos = Gf.Vec3d(*world_pos)

    xformable = UsdGeom.Xformable(prim)
    translate_op = None
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
            break
    if translate_op is None:
        xformable.ClearXformOpOrder()
        translate_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    translate_op.Set(local_pos)


def _bind_preview_surface(stage, target_prim, material_path: str, color: tuple[float, float, float], opacity: float) -> None:
    if UsdShade is None or Sdf is None:
        return
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(target_prim).Bind(material)


def create_detection_zone_visualization(
    env: ManagerBasedRLEnv | DirectRLEnv,
    container_cfg: SceneEntityCfg,
    container_name: str,
    x_range: tuple[float, float] = (-0.10, 0.10),
    y_range: tuple[float, float] = (-0.10, 0.10),
    z_range: tuple[float, float] = (-0.05, 0.05),
    color: tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> bool:
    if not USD_AVAILABLE:
        return False

    try:
        stage = env.sim.stage
        container: RigidObject = env.scene[container_cfg.name]
    except Exception:
        return False

    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range
    box_size = (x_max - x_min, y_max - y_min, z_max - z_min)
    box_center = ((x_min + x_max) * 0.5, (y_min + y_max) * 0.5, (z_min + z_max) * 0.5)

    env_base_path = _get_env_base_path(env, container_cfg.name)
    container_pos, _ = get_object_pose_w(container, env_id=None, prefer_link_frame=True)

    for env_id in range(env.num_envs):
        pos = container_pos[env_id].cpu()
        zone_world_pos = (float(pos[0] + box_center[0]), float(pos[1] + box_center[1]), float(pos[2] + box_center[2]))

        env_path = env_base_path.replace("env_0", f"env_{env_id}")
        zone_path = f"{env_path}/Scene/DetectionZone_{container_name}"
        if stage.GetPrimAtPath(zone_path).IsValid():
            stage.RemovePrim(zone_path)

        zone_xform = UsdGeom.Xform.Define(stage, zone_path).GetPrim()
        _set_or_create_translate_op_from_world(zone_xform, zone_world_pos)

        cube = UsdGeom.Cube.Define(stage, f"{zone_path}/Cube")
        cube.GetSizeAttr().Set(1.0)
        cube.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])

        cube_xformable = UsdGeom.Xformable(cube.GetPrim())
        cube_xformable.ClearXformOpOrder()
        cube_xformable.AddScaleOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(*box_size))

        try:
            _bind_preview_surface(stage, cube.GetPrim(), f"{zone_path}/Material", color, opacity=0.25)
        except Exception:
            pass

    return True


def create_object_detection_point_visualization(
    env: ManagerBasedRLEnv | DirectRLEnv,
    object_cfg: SceneEntityCfg,
    object_name: str,
    object_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    offset_frame: str = "local",
    color: tuple[float, float, float] = (1.0, 0.84, 0.0),
    radius: float = 0.015,
) -> bool:
    if not USD_AVAILABLE:
        return False

    try:
        stage = env.sim.stage
        object_entity: RigidObject = env.scene[object_cfg.name]
    except Exception:
        return False

    env_base_path = _get_env_base_path(env, object_cfg.name)
    use_offset_frame = "local" if offset_frame == "local" else "world"
    prim_paths: list[str] = []

    for env_id in range(env.num_envs):
        point_world_pos = _compute_detection_point_world_pos(object_entity, env_id, object_offset, use_offset_frame)
        env_path = env_base_path.replace("env_0", f"env_{env_id}")
        point_path = f"{env_path}/Scene/DetectionPoint_{object_name}"
        prim_paths.append(point_path)

        if stage.GetPrimAtPath(point_path).IsValid():
            stage.RemovePrim(point_path)

        point_xform = UsdGeom.Xform.Define(stage, point_path).GetPrim()
        _set_or_create_translate_op_from_world(point_xform, point_world_pos)

        sphere = UsdGeom.Sphere.Define(stage, f"{point_path}/Sphere")
        sphere.GetRadiusAttr().Set(radius)
        sphere.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])

        try:
            _bind_preview_surface(stage, sphere.GetPrim(), f"{point_path}/Material", color, opacity=1.0)
        except Exception:
            pass

    if not hasattr(env, "_task1_detection_point_visualizations"):
        env._task1_detection_point_visualizations = {}

    env._task1_detection_point_visualizations[object_name] = {
        "object_cfg": object_cfg,
        "object_offset": object_offset,
        "offset_frame": use_offset_frame,
        "prim_paths": prim_paths,
    }
    return True


def update_object_detection_point_visualization(
    env: ManagerBasedRLEnv | DirectRLEnv,
    object_name: str,
) -> None:
    if not USD_AVAILABLE or not hasattr(env, "_task1_detection_point_visualizations"):
        return
    if object_name not in env._task1_detection_point_visualizations:
        return

    vis_info = env._task1_detection_point_visualizations[object_name]
    object_entity: RigidObject = env.scene[vis_info["object_cfg"].name]
    object_offset = vis_info["object_offset"]
    offset_frame = vis_info.get("offset_frame", "local")
    stage = env.sim.stage

    for env_id, prim_path in enumerate(vis_info["prim_paths"]):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
        point_world_pos = _compute_detection_point_world_pos(object_entity, env_id, object_offset, offset_frame)
        _set_or_create_translate_op_from_world(prim, point_world_pos)


def init_task1_visualization(
    env: ManagerBasedRLEnv | DirectRLEnv,
    plate_name: str,
    object_names: tuple[str, ...],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    height_range: tuple[float, float],
    object_offsets: dict[str, tuple[float, float, float]] | None = None,
    object_regions: dict[str, dict[str, tuple[float, float]]] | None = None,
    offset_frame: str = "local",
) -> None:
    if hasattr(env, "_task1_visualization_created"):
        return

    offsets = object_offsets or {}
    use_offset_frame = "local" if offset_frame == "local" else "world"

    if object_regions:
        zone_colors = (
            (0.0, 1.0, 0.0),
            (0.1, 0.7, 1.0),
            (1.0, 0.4, 0.0),
            (1.0, 0.8, 0.2),
        )
        for index, (region_name, region_cfg) in enumerate(object_regions.items()):
            region_x_range = region_cfg.get("x_range", x_range)
            region_y_range = region_cfg.get("y_range", y_range)
            region_h_range = region_cfg.get("height_range", height_range)
            create_detection_zone_visualization(
                env=env,
                container_cfg=SceneEntityCfg(plate_name),
                container_name=f"{plate_name}_{region_name}",
                x_range=region_x_range,
                y_range=region_y_range,
                z_range=region_h_range,
                color=zone_colors[index % len(zone_colors)],
            )
    else:
        create_detection_zone_visualization(
            env=env,
            container_cfg=SceneEntityCfg(plate_name),
            container_name=plate_name,
            x_range=x_range,
            y_range=y_range,
            z_range=height_range,
            color=(0.0, 1.0, 0.0),
        )

    for object_name in object_names:
        create_object_detection_point_visualization(
            env=env,
            object_cfg=SceneEntityCfg(object_name),
            object_name=object_name,
            object_offset=offsets.get(object_name, (0.0, 0.0, 0.0)),
            offset_frame=use_offset_frame,
            color=(1.0, 0.84, 0.0),
            radius=0.012,
        )

    env._task1_visualization_created = True


def update_task1_visualization(env: ManagerBasedRLEnv | DirectRLEnv) -> None:
    if not hasattr(env, "_task1_detection_point_visualizations"):
        return
    for object_name in tuple(env._task1_detection_point_visualizations.keys()):
        update_object_detection_point_visualization(env, object_name)
