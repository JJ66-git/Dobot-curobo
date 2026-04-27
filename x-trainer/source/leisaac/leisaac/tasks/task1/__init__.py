import gymnasium as gym
from .xtrainer_pickup_recognition_env_cfg import Task1EnvCfg

gym.register(
    id="task1",  # 新任务ID
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Task1EnvCfg,
    },
)
