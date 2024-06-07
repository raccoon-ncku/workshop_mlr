import gymnasium as gym
import raccoon_gym

env = gym.make('RaccoonKr300R2500UltraReach-v1', render_mode="human")

observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample() # random action
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()