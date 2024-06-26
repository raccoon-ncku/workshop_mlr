from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.callbacks import CheckpointCallback
import gymnasium as gym
import panda_gym
import pathlib
import os
from datetime import datetime

# Environment and Training settings
TOTAL_TIMESTEPS=1_000_000
ENV = "PandaReach-v3"
MODEL_CLASS = DDPG  # works also with SAC, DDPG and TD3
MODEL_CLASS_NAME = "ddpg"
# Save Settings
CWD = pathlib.Path(os.getcwd())
SAVE_DIR = CWD / "logs" / MODEL_CLASS_NAME / (ENV + datetime.now().strftime("%y%m%dT%H%M"))
OUTPUT_FILE = SAVE_DIR / "rl_model"


env = gym.make("PandaPush-v3")
# Available strategies (cf paper): future, final, episode
goal_selection_strategy = "future" # equivalent to GoalSelectionStrategy.FUTURE

# Initialize the model
model = MODEL_CLASS(
    policy="MultiInputPolicy",
    env=env,
    replay_buffer_class=HerReplayBuffer,
    # Parameters for HER
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy=goal_selection_strategy,
    ),
    verbose=1,
    tensorboard_log=SAVE_DIR / "tensorboard"
)

checkpoint_callback = CheckpointCallback( 
    save_freq=50_000,
    save_path=SAVE_DIR, 
    name_prefix=MODEL_CLASS_NAME + "-" + ENV 
)  # Callback for saving the model

# Save the model
try:
    model.learn(TOTAL_TIMESTEPS, checkpoint_callback, progress_bar=True)
except KeyboardInterrupt:
    pass
model.save(OUTPUT_FILE)
print("Model saved")
env.close()