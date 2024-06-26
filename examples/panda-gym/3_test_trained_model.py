import gymnasium as gym
import panda_gym
from stable_baselines3 import DDPG
import time

env = gym.make("PandaReach-v3", render_mode="human")
model = DDPG(policy="MultiInputPolicy", env=env, verbose=1)

vec_env = model.get_env()

del model # remove to demonstrate saving and loading

model = DDPG.load("logs/ddpg/PandaReach-v3240626T1100/rl_model.zip")

obs = vec_env.reset()

while True:
    try:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        env.render()
        time.sleep(1/60)
    except KeyboardInterrupt:
        break

env.close()
