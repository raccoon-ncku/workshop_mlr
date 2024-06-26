import gymnasium as gym
import panda_gym
from stable_baselines3 import DDPG
from datetime import datetime
import pathlib
import os
from stable_baselines3.common.callbacks import CheckpointCallback

# This script contains a callback to save the model every 5,000 timesteps
# and provides a tensorboard log for visualization
# to access tensorboard, run the following command in the terminal
# tensorboard --logdir <PATH_TO_TENSORBOARD> --host 0.0.0.0

# Environment and Training settings
TOTAL_TIMESTEPS=30_000
ENV = "PandaReach-v3"

# Save Settings
CWD = pathlib.Path(os.getcwd())
SAVE_DIR = CWD / "logs" / "ddpg" / (ENV + datetime.now().strftime("%y%m%dT%H%M"))
OUTPUT_FILE = SAVE_DIR / "rl_model"

env = gym.make("PandaReach-v3")
model = DDPG(policy="MultiInputPolicy", env=env, verbose=1, tensorboard_log=SAVE_DIR / "tensorboard")

checkpoint_callback = CheckpointCallback( 
    save_freq=5_000,
    save_path=SAVE_DIR, 
    name_prefix="DDPG-"+ENV 
)  # Callback for saving the model

# Save the model
try:
    model.learn(TOTAL_TIMESTEPS, checkpoint_callback, progress_bar=True)
except KeyboardInterrupt:
    pass
model.save(OUTPUT_FILE)
print("Model saved")
env.close()
