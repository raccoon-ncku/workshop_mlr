import gymnasium as gym
import panda_gym
from stable_baselines3 import DDPG
from datetime import datetime
import pathlib
import os

# Environment and Training settings
TOTAL_TIMESTEPS=30_000
ENV = "PandaReach-v3"

# Save Settings
CWD = pathlib.Path(os.getcwd())
SAVE_DIR = CWD / "logs" / "ddpg" / (ENV + datetime.now().strftime("%y%m%dT%H%M"))
OUTPUT_FILE = SAVE_DIR / "rl_model"

env = gym.make("PandaReach-v3")
model = DDPG(policy="MultiInputPolicy", env=env, verbose=1)

# Save the model
try:
    model.learn(TOTAL_TIMESTEPS, progress_bar=True)
except KeyboardInterrupt:
    pass
model.save(OUTPUT_FILE)
print("Model saved")
env.close()
