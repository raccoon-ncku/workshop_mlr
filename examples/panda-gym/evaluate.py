import gymnasium as gym
from stable_baselines3 import DDPG
from stable_baselines3.common.evaluation import evaluate_policy
import panda_gym

env = gym.make("PandaPickAndPlace-v3")
model = DDPG("MultiInputPolicy", env)
evaluate_policy(model, env, n_eval_episodes=100, warn=False)
env.close()