from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3
from sb3_contrib import TQC
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.envs import BitFlippingEnv
import gymnasium as gym
from datetime import datetime
import raccoon_gym

TOTAL_TIMESTEPS=100_000
ENV = "RaccoonKr16Reach-v1"
OUTPUT_FILE = "{}-ddpg-{}".format(ENV, datetime.now().strftime("%y%m%dT%H%M"))
BASE_PATH = "output"
TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
model_class = TQC  # works also with SAC, DDPG and TD3

# env = BitFlippingEnv(n_bits=N_BITS, continuous=model_class in [DDPG, SAC, TD3], max_steps=N_BITS)
env = gym.make("RaccoonKr16Reach-v1", control_type="joints", render_mode="human", renderer="Tiny")
# Available strategies (cf paper): future, final, episode
goal_selection_strategy = "future" # equivalent to GoalSelectionStrategy.FUTURE

# Initialize the model
model = TQC(
    "MultiInputPolicy",
    env,
    batch_size=2048,
    buffer_size=1000000,
    gamma=0.95,
    learning_rate=0.001,
    policy_kwargs=dict(net_arch=[512, 512, 512], n_critics=2),
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(goal_selection_strategy='future', n_sampled_goal=4),
    tau=0.05,
    seed=3157870761,
    verbose=1,
    tensorboard_log=f'{BASE_PATH}/tensorboard/',
)

model = model_class.load("logs/models/20240627102311/tqc_kuka_reach_30000_steps.zip", env=env)

obs, info = env.reset()
# for _ in range(100):
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, _ = env.step(action)
#     if terminated or truncated:
#         obs, info = env.reset()

# env.close()
import time
while True:
    try:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        time.sleep(0.1)
        if terminated or truncated:
            obs, info = env.reset()
    except KeyboardInterrupt:
        break

env.close()