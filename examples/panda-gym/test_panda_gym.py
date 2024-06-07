import gymnasium as gym
import panda_gym
from stable_baselines3 import DDPG, HerReplayBuffer

env = gym.make("PandaFlip-v3", render_mode="human")
model = DDPG(policy="MultiInputPolicy", replay_buffer_class=HerReplayBuffer, env=env, verbose=1)

vec_env = model.get_env()

del model # remove to demonstrate saving and loading

model = DDPG.load("ddpg_pandapush.zip")

obs = vec_env.reset()

import time
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    env.render()
