import gymnasium as gym
import raccoon_gym
import numpy as np
from stable_baselines3 import DDPG
from datetime import datetime
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import TQC
import pathlib
import os
TOTAL_TIMESTEPS=100_000
ENV = "RaccoonKr3Reach-v1"
OUTPUT_FILE = "{}-ddpg-{}".format(ENV, datetime.now().strftime("%y%m%dT%H%M"))
BASE_PATH = "logs"
TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")

env = gym.make(ENV, control_type="joints")
# model = DDPG(policy="MultiInputPolicy", env=env, verbose=1)

model = TQC(
    "MultiInputPolicy",
    env,
    batch_size=2048,
    buffer_size=1000000,
    gamma=0.95,
    learning_rate=0.001,
    policy_kwargs=dict(net_arch=[512, 512, 512], n_critics=2),
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(goal_selection_strategy='future', n_sampled_goal=4),
    tau=0.05,
    seed=3157870761,
    verbose=1,
    tensorboard_log=f'{BASE_PATH}/tensorboard/',
)

checkpoint_callback = CheckpointCallback( 
    save_freq=30_000,
    save_path=f"{BASE_PATH}/models/{TIMESTAMP}/", 
    name_prefix="tqc_kuka_reach"
)  # Callback for saving the model

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=checkpoint_callback, 
    progress_bar=True
)

# Save the model
model.save(f"{BASE_PATH}/kr3_reach")
print("Model saved")
env.close()
