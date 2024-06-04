import gymnasium as gym
import panda_gym
from stable_baselines3 import DDPG
from datetime import datetime

from argparse import ArgumentParser, RawTextHelpFormatter

# Define mapping of short names to full environment names
ENV_NAMES = {
    'reach': 'PandaReach-v3',
    'push': 'PandaPush-v3',
    'slide': 'PandaSlide-v3',
    'pick': 'PandaPickAndPlace-v3',
    'stack': 'PandaStack-v3',
    'flip': 'PandaFlip-v3'
}

# Generate help string for --env argument
env_help = ['{}: {}'.format(a, b) for a, b in zip(ENV_NAMES.keys(), ENV_NAMES.values())]
env_help_string = "The Gym environment to use. Available short names: \n" + '\n'.join(env_help)

# Set up argument parser
parser = ArgumentParser(description='Train a DDPG model with specified environment and policy.', formatter_class=RawTextHelpFormatter)
parser.add_argument('--env', type=str, default='reach', help=env_help_string)
parser.add_argument('--policy', type=str, default='MultiInputPolicy', help='The policy to use for the DDPG model')
parser.add_argument('--timesteps', type=int, default=30_000, help='The total number of timesteps to train the model')

# Parse arguments
args = parser.parse_args()
env_name = ENV_NAMES[args.env]
# Set up output file
OUTPUT_FILE = "{}-ddpg-{}".format(env_name, datetime.now().strftime("%y%m%dT%H%M"))

# Create environment and model
env = gym.make(env_name)
model = DDPG(policy=args.policy, env=env, verbose=1)

# Train the model
model.learn(args.timesteps, progress_bar=True)

# Save the model
model.save(OUTPUT_FILE)
print("Model saved")
