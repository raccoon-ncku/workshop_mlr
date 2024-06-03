
import gymnasium as gym
import stable_baselines3 as sb3
import matplotlib.pyplot as plt
import cv2
import numpy as np


# We then create the pendulum environment built into Gymnasium.
tmp_env = gym.make('Pendulum-v1', render_mode='rgb_array')

# wrap the env in the record video
env = gym.wrappers.RecordVideo(env=tmp_env, video_folder="", name_prefix="test-video", episode_trigger=lambda x: x % 2 == 0)

# env reset for a fresh start
observation, info = env.reset()

###
# Start the recorder
env.start_video_recorder()


# We create a function to test the given agent (“model”) in our environment. Note that this resets the environment and runs forever until terminated or truncated comes back False. Our model is used to predict an action from a given observation (known as the policy). This action is used to take a step (with the .step() function) in the environment, which returns a new observation, reward, and whether or not the environment has terminated/truncated.
# If a video handle (from OpenCV) is passed in, each step is rendered and added to the video. If a message (msg) is passed in, that text will appear on the top-left of the video.
def test_model(env: gym.Env, model, video=None, msg=None):
    # Reset environment
    obs, info = env.reset()
    frame = env.render()
    ep_len = 0
    ep_rew = 0

    # Run episode until complete
    while True:
        # Provide observation to policy to predict the next action
        action, _ = model.predict(obs)

        # Perform action, update total reward
        obs, reward, terminated, truncated, info = env.step(action)
        ep_rew += reward

        # Record frame to video
        if video:
            frame = env.render()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.putText(
                frame,                    # Image
                msg,                      # Text to add
                (10, 25),                 # Origin of text in imagg
                cv2.FONT_HERSHEY_SIMPLEX, # Font
                1,                        # Font scale
                (0, 0, 0,),               # Color
                2,                        # Thickness
                cv2.LINE_AA               # Line type
            )
            video.write(frame)

        # Increase step counter
        ep_len += 1

        # Check to see if episode has ended
        if terminated or truncated:
            return ep_len, ep_rew


# From there, we create a dummy agent that simply selects actions randomly.
# Model that just predicts random actions
class DummyModel():
    # Save environment
    def __init__(self, env):
        self.env = env

    # Always output random action regardless of observation
    def predict(self, obs):
        action = self.env.action_space.sample()
        return action, None


# We then configure our video writer object and run a few episodes with our random agent. Feel free to download the output video (1-random.mp4) to see how this agent performs (probably poorly).
# Recorder settings
FPS = 30
FOURCC = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
VIDEO_FILENAME = "1-random.mp4"

# Use frame from environment to compute resolution
width = 640
height = 480

# Create recorder
video = cv2.VideoWriter(VIDEO_FILENAME, FOURCC, FPS, (width, height))

# Try running a few episodes with the environment and random actions
dummy_model = DummyModel(env)
for ep in range(5):
    ep_len, ep_rew = test_model(env, dummy_model, video, f"Random, episode {ep}")
    print(f"Episode {ep} | length: {ep_len}, reward: {ep_rew}")

# Close the video writer
video.release()
env.close_video_recorder()

# Next, we initialize our model (agent) to train with the PPO algorithm and set some hyperparameters. As noted, our hyperparameters come from the rl-baseline3-zoo repository.
# Initialize model
model = sb3.PPO(
    'MlpPolicy',
    env,
    learning_rate=0.001,       # Learning rate of neural network (default: 0.0003)
    n_steps=1024,               # Number of steps per update (default: 2048)
    batch_size=64,              # Minibatch size for NN update (default: 64)
    gamma=0.9,                 # Discount factor (default: 0.99)
    ent_coef=0.0,               # Entropy, how much to explore (default: 0.0)
    use_sde=True,               # Use generalized State Dependent Exploration (default: False)
    sde_sample_freq=4,          # Number of steps before sampling new noise matrix (default -1)
    policy_kwargs={'net_arch': [64, 64]}, # 2 hidden layers, 1 output layer (default: [64, 64])
    verbose=0                   # Print training metrics (default: 0)
)


# With our model configured, we train. To make our demo more visually appealing, we divide the training into “rounds.” In each round, we train for a given number of steps and then test the model in our environment 100 times. The first test is recorded to video, and the episode lengths and rewards are averaged over all 100 tests.
# This can take 5-10 minutes. Once done, you can download the output video (2-training.mp4) to see the training progress.
# Training and testing hyperparameters
NUM_ROUNDS = 2
NUM_TRAINING_STEPS_PER_ROUND = 5000
NUM_TESTS_PER_ROUND = 100
MODEL_FILENAME_BASE = "pendulum-ppo"
VIDEO_FILENAME = "2-training.mp4"

# Create recorder
video = cv2.VideoWriter(VIDEO_FILENAME, FOURCC, FPS, (width, height))

# Train and test the model for a number of rounds
avg_ep_lens = []
avg_ep_rews = []
for rnd in range(NUM_ROUNDS):
    # Train the model
    model.learn(total_timesteps=NUM_TRAINING_STEPS_PER_ROUND)

    # Save the model
    model.save(f"{MODEL_FILENAME_BASE}_{rnd}")

    # Test the model in several episodes
    avg_ep_len = 0
    avg_ep_rew = 0
    for ep in range(NUM_TESTS_PER_ROUND):

        # Only record the first test
        if ep == 0:
            ep_len, ep_rew = test_model(env, model, video, f"Round {rnd}")
        else:
            ep_len, ep_rew = test_model(env, model)

        # Accumulate average length and reward
        avg_ep_len += ep_len
        avg_ep_rew += ep_rew

    # Record and dieplay average episode length and reward
    avg_ep_len /= NUM_TESTS_PER_ROUND
    avg_ep_lens.append(avg_ep_len)
    avg_ep_rew /= NUM_TESTS_PER_ROUND
    avg_ep_rews.append(avg_ep_rew)
    print(f"Round {rnd} | average test length: {avg_ep_len}, average test reward: {avg_ep_rew}")

# Close the video writer
video.release()


# Plot average test episode lengths and rewards for each round
fig, axs = plt.subplots(1, 2)
fig.tight_layout(pad=4.0)
axs[0].plot(avg_ep_lens)
axs[0].set_ylabel("Average episode length")
axs[0].set_xlabel("Round")
axs[1].plot(avg_ep_rews)
axs[1].set_ylabel("Average episode reward")
axs[1].set_xlabel("Round")


# hoose a model that provided the best average rewards, load it, and call our test_model() function using that model. For example, in the plot above, we can see that the model from round 17 gave the best average test results. So, change the MODEL_FILENAME value to “pendulum-ppo_17” and run the following cell.
# Model and video settings
MODEL_FILENAME = "pendulum-ppo_1"
VIDEO_FILENAME = "3-testing.mp4"

# Load the model
model = sb3.PPO.load(MODEL_FILENAME)

# Create recorder
video = cv2.VideoWriter(VIDEO_FILENAME, FOURCC, FPS, (width, height))

# Test the model
ep_len, ep_rew = test_model(env, model, video, MODEL_FILENAME)
print(f"Episode length: {ep_len}, reward: {ep_rew}")

# Close the video writer
video.release()


# Close your environment.
env.close()
