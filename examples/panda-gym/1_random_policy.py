import gymnasium as gym
import panda_gym

# Create the environment
# other environments:
# PandaReach-v3
# PandaPush-v3
# PandaSlide-v3
# PandaPickAndPlace-v3
# PandaStack-v3
# PandaFlip-v3
env = gym.make('PandaReach-v3', render_mode="human")

observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample() # random action
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()