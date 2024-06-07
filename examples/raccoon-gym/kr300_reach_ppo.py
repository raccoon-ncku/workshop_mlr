import gymnasium as gym
import raccoon_gym
from stable_baselines3 import DDPG
from datetime import datetime

TOTAL_TIMESTEPS=30_000
ENV = "RaccoonKr300R2500UltraReach-v1"
OUTPUT_FILE = "{}-ddpg-{}".format(ENV, datetime.now().strftime("%y%m%dT%H%M"))

env = gym.make(ENV)
model = DDPG(policy="MultiInputPolicy", env=env, verbose=1)
model.learn(TOTAL_TIMESTEPS, progress_bar=True)

# Save the model
model.save(OUTPUT_FILE)
print("Model saved")
env.close()
