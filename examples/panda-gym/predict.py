import gymnasium as gym
import panda_gym
from stable_baselines3 import DDPG

env = gym.make("PandaReach-v3", render_mode="human")
model = DDPG(policy="MultiInputPolicy", env=env, verbose=1)

vec_env = model.get_env()

del model # remove to demonstrate saving and loading
print("Model deleted")
model = DDPG.load("ddpg_panda")
print("Model loaded")

obs = vec_env.reset()
print("Model loaded")
print(env)
import time
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    time.sleep(0.1)
    env.render()
