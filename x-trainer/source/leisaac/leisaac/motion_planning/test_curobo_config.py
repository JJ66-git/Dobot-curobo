"""Tests for resolving project-local cuRobo robot configuration."""

from pathlib import Path
from types import SimpleNamespace

from leisaac.motion_planning.curobo_config import (
    build_curobo_robot_config,
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


def test_build_curobo_robot_config_uses_absolute_project_paths(monkeypatch) -> None:
    observed = {}

    class FakeContentPath:
        def __init__(
            self,
            robot_config_absolute_path=None,
            robot_urdf_absolute_path=None,
            robot_asset_absolute_path=None,
        ) -> None:
            observed["content_path"] = {
                "robot_config_absolute_path": robot_config_absolute_path,
                "robot_urdf_absolute_path": robot_urdf_absolute_path,
                "robot_asset_absolute_path": robot_asset_absolute_path,
            }
            self.robot_config_absolute_path = robot_config_absolute_path
            self.robot_urdf_absolute_path = robot_urdf_absolute_path
            self.robot_asset_absolute_path = robot_asset_absolute_path

    def fake_load_robot_yaml(content_path):
        observed["loaded_content_path"] = content_path
        return {
            "robot_cfg": {
                "kinematics": {
                    "urdf_path": content_path.robot_urdf_absolute_path,
                    "asset_root_path": content_path.robot_asset_absolute_path,
                    "collision_spheres": str(
                        Path(content_path.robot_asset_absolute_path)
                        / "xtrainer_spheres.yml"
                    ),
                    "base_link": "base_link",
                    "tool_frames": ["left_ee_link", "right_ee_link"],
                }
            }
        }

    class FakeRobotCfg:
        @staticmethod
        def create(robot_data):
            observed["robot_data"] = robot_data
            return SimpleNamespace(
                kinematics=SimpleNamespace(
                    generator_config=SimpleNamespace(
                        urdf_path=robot_data["robot_cfg"]["kinematics"]["urdf_path"],
                        asset_root_path=robot_data["robot_cfg"]["kinematics"]["asset_root_path"],
                    )
                )
            )

    monkeypatch.setattr(
        "leisaac.motion_planning.curobo_config._import_curobo_robot_config_helpers",
        lambda: (FakeContentPath, fake_load_robot_yaml, FakeRobotCfg),
    )

    resolved = build_curobo_robot_config("xtrainer.yml")
    generator_config = resolved.kinematics.generator_config

    assert Path(observed["content_path"]["robot_config_absolute_path"]).name == "xtrainer.yml"
    assert Path(observed["content_path"]["robot_config_absolute_path"]).is_absolute()
    assert Path(observed["content_path"]["robot_urdf_absolute_path"]).name == "xtrainer.urdf"
    assert Path(observed["content_path"]["robot_urdf_absolute_path"]).is_absolute()
    assert Path(observed["content_path"]["robot_asset_absolute_path"]).name == "my_x_trainer"
    assert Path(observed["content_path"]["robot_asset_absolute_path"]).is_absolute()
    assert Path(generator_config.urdf_path).is_absolute()
    assert Path(generator_config.asset_root_path).is_absolute()
    assert observed["robot_data"]["robot_cfg"]["kinematics"]["urdf_path"] == observed["content_path"][
        "robot_urdf_absolute_path"
    ]


if __name__ == "__main__":
    test_get_project_xtrainer_config_path()
    test_resolve_robot_config_input_for_project_default()
    test_resolve_robot_config_input_preserves_builtin_robot_name()
    print("test_curobo_config: base checks PASS (run with pytest for full suite)")
