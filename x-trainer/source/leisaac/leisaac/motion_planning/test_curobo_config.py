"""Tests for resolving project-local cuRobo robot configuration."""

from pathlib import Path

from leisaac.motion_planning.curobo_config import (
    get_project_xtrainer_config_path,
    resolve_robot_config_input,
)


def test_get_project_xtrainer_config_path() -> None:
    config_path = get_project_xtrainer_config_path()

    assert config_path.name == "xtrainer.yml"
    assert config_path.is_file()
    assert config_path.parent.name == "my_x_trainer"


def test_resolve_robot_config_input_for_project_default() -> None:
    resolved = resolve_robot_config_input("xtrainer.yml")

    assert isinstance(resolved, dict)
    kinematics = resolved["robot_cfg"]["kinematics"]

    assert Path(kinematics["urdf_path"]).is_absolute()
    assert Path(kinematics["urdf_path"]).name == "xtrainer.urdf"
    assert Path(kinematics["asset_root_path"]).is_absolute()
    assert Path(kinematics["asset_root_path"]).name == "my_x_trainer"


def test_resolve_robot_config_input_preserves_builtin_robot_name() -> None:
    resolved = resolve_robot_config_input("franka.yml")

    assert resolved == "franka.yml"


if __name__ == "__main__":
    test_get_project_xtrainer_config_path()
    test_resolve_robot_config_input_for_project_default()
    test_resolve_robot_config_input_preserves_builtin_robot_name()
    print("test_curobo_config: PASS")
