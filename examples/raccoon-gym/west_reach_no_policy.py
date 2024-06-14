import gymnasium as gym
import raccoon_gym
import time


env = gym.make('RCCNWestRobotReach-v1', render_mode="human")

observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample() # random action
    observation, reward, terminated, truncated, info = env.step(action)
    time.sleep(0.1)

    if terminated or truncated:
        observation, info = env.reset()