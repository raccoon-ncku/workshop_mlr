import os

from gymnasium.envs.registration import register

__version__ = "0.0.1"

ENV_IDS = []

# KUKA KR300 R2500 Ultra
for task in ["Reach"]:
    for reward_type in ["sparse", "dense"]:
        for control_type in ["ee", "joints"]:
            reward_suffix = "Dense" if reward_type == "dense" else ""
            control_suffix = "Joints" if control_type == "joints" else ""
            env_id = f"RaccoonKr300R2500Ultra{task}{control_suffix}{reward_suffix}-v1"

            register(
                id=env_id,
                entry_point=f"raccoon_gym.envs:Kr300R2500Ultra{task}Env",
                kwargs={"reward_type": reward_type, "control_type": control_type},
                max_episode_steps=100
            )

            ENV_IDS.append(env_id)

# Raccoon West RCCNWestRobotReachEnv
for task in ["Reach"]:
    for reward_type in ["sparse", "dense"]:
        for control_type in ["ee", "joints"]:
            reward_suffix = "Dense" if reward_type == "dense" else ""
            control_suffix = "Joints" if control_type == "joints" else ""
            env_id = f"RCCNWestRobot{task}{control_suffix}{reward_suffix}-v1"

            register(
                id=env_id,
                entry_point=f"raccoon_gym.envs:RCCNWestRobot{task}Env",
                kwargs={"reward_type": reward_type, "control_type": control_type},
                max_episode_steps=100
            )

            ENV_IDS.append(env_id)
