"""Helpers for resolving project-local cuRobo robot configuration."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Union


RobotConfigInput = Union[str, Dict[str, Any]]

_PROJECT_XTRAINER_CONFIG = Path("curobo") / "my_x_trainer" / "xtrainer.yml"


def get_project_xtrainer_config_path() -> Path:
    """Return the repo-local X-Trainer config path."""
    search_start = Path(__file__).resolve().parent
    for candidate in (search_start, *search_start.parents):
        config_path = candidate / _PROJECT_XTRAINER_CONFIG
        if config_path.is_file():
            return config_path
    raise FileNotFoundError(
        "Could not locate project-local xtrainer.yml under curobo/my_x_trainer. "
        "Please check the repository layout."
    )


def resolve_robot_config_input(robot_config: RobotConfigInput) -> RobotConfigInput:
    """Resolve repo-local X-Trainer config into a dict with absolute asset paths."""
    if isinstance(robot_config, dict):
        return _normalize_robot_config_dict(robot_config)

    config_path = _resolve_config_path(robot_config)
    if config_path is None:
        return robot_config
    return _load_robot_config(config_path)


def _resolve_config_path(robot_config: str) -> Path | None:
    candidate = Path(robot_config)
    if candidate.is_absolute() or candidate.exists():
        return candidate.resolve()
    if candidate.name == "xtrainer.yml":
        return get_project_xtrainer_config_path()
    return None


def _load_robot_config(config_path: Path) -> Dict[str, Any]:
    import yaml

    with config_path.open("r", encoding="utf-8") as handle:
        robot_config = yaml.safe_load(handle)
    return _normalize_robot_config_dict(robot_config, config_path)


def _normalize_robot_config_dict(
    robot_config: Dict[str, Any], config_path: Path | None = None
) -> Dict[str, Any]:
    normalized = copy.deepcopy(robot_config)

    robot_cfg = normalized.get("robot_cfg")
    if not isinstance(robot_cfg, dict):
        return normalized

    kinematics = robot_cfg.get("kinematics")
    if not isinstance(kinematics, dict):
        return normalized

    if config_path is None:
        return normalized

    config_dir = config_path.resolve().parent
    kinematics["asset_root_path"] = str(
        _make_absolute_path(kinematics.get("asset_root_path", "."), config_dir)
    )

    urdf_path = kinematics.get("urdf_path")
    if isinstance(urdf_path, str) and urdf_path:
        kinematics["urdf_path"] = str(_make_absolute_path(urdf_path, config_dir))

    collision_spheres = kinematics.get("collision_spheres")
    if isinstance(collision_spheres, str) and collision_spheres:
        kinematics["collision_spheres"] = str(
            _make_absolute_path(collision_spheres, config_dir)
        )

    return normalized


def _make_absolute_path(raw_path: str, base_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()
