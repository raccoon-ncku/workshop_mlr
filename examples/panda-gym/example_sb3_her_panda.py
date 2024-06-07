from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
import gymnasium as gym
import panda_gym


model_class = DDPG  # works also with SAC, DDPG and TD3

# env = BitFlippingEnv(n_bits=N_BITS, continuous=model_class in [DDPG, SAC, TD3], max_steps=N_BITS)
env = gym.make("PandaPush-v3")
# Available strategies (cf paper): future, final, episode
goal_selection_strategy = "future" # equivalent to GoalSelectionStrategy.FUTURE

# Initialize the model
model = model_class(
    policy="MultiInputPolicy",
    env=env,
    replay_buffer_class=HerReplayBuffer,
    # Parameters for HER
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy=goal_selection_strategy,
    ),
    verbose=1
)


# Train the model
model.learn(1000000, progress_bar=True)

model.save("./her_panda_env")
env.close()
# Because it needs access to `env.compute_reward()`
# HER must be loaded with the env
# model = model_class.load("./her_bit_env", env=env)

# obs, info = env.reset()
# for _ in range(100):
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, _ = env.step(action)
#     if terminated or truncated:
#         obs, info = env.reset()