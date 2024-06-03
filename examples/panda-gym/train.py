import gymnasium as gym
import panda_gym
from stable_baselines3 import DDPG

env = gym.make("PandaReach-v3")
model = DDPG(policy="MultiInputPolicy", env=env, verbose=1)
model.learn(30_000, progress_bar=True)


# Save the model
model.save("ddpg_panda")
print("Model saved")
