# %% [markdown]
# This Jupyter Notebook is a document that contains multiple cells of code and markdown. It demonstrates the process of training and testing a model using the PPO algorithm in the Gymnasium environment. The notebook includes the installation of necessary packages, the creation of the environment, the definition of a dummy model, the training and testing of the model, and the visualization of the results. It also includes the use of OpenCV for video recording and rendering. The notebook showcases the use of various variables, modules, and functions to perform the desired tasks.

# %%
import gymnasium as gym
import stable_baselines3 as sb3
import matplotlib.pyplot as plt
import numpy as np

# Check versions
print(f"gym version: {gym.__version__}")

# %% [markdown]
# We then create the pendulum environment built into Gymnasium.

# %%
env = gym.make('Pendulum-v1', render_mode='rgb_array')

# %% [markdown]
# We create a function to test the given agent (“model”) in our environment. Note that this resets the environment and runs forever until terminated or truncated comes back False. Our model is used to predict an action from a given observation (known as the policy). This action is used to take a step (with the .step() function) in the environment, which returns a new observation, reward, and whether or not the environment has terminated/truncated.
# 
# If a video handle (from OpenCV) is passed in, each step is rendered and added to the video. If a message (msg) is passed in, that text will appear on the top-left of the video.

# %%
def test_model(env: gym.Env, model, msg=None):

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

        # Increase step counter
        ep_len += 1

        # Check to see if episode has ended
        if terminated or truncated:
            return ep_len, ep_rew

# %% [markdown]
# From there, we create a dummy agent that simply selects actions randomly.

# %%
# Model that just predicts random actions
class DummyModel():

    # Save environment
    def __init__(self, env):
        self.env = env

    # Always output random action regardless of observation
    def predict(self, obs):
        action = self.env.action_space.sample()
        return action, None

# %% [markdown]
# We then configure our video writer object and run a few episodes with our random agent. Feel free to download the output video (1-random.mp4) to see how this agent performs (probably poorly).
# 
# 

# %%

# Try running a few episodes with the environment and random actions
dummy_model = DummyModel(env)
for ep in range(5):
    ep_len, ep_rew = test_model(env, dummy_model, f"Random, episode {ep}")
    print(f"Episode {ep} | length: {ep_len}, reward: {ep_rew}")

# %% [markdown]
# Next, we initialize our model (agent) to train with the PPO algorithm and set some hyperparameters. As noted, our hyperparameters come from the rl-baseline3-zoo repository.

# %%
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

# %% [markdown]
# With our model configured, we train. To make our demo more visually appealing, we divide the training into “rounds.” In each round, we train for a given number of steps and then test the model in our environment 100 times. The first test is recorded to video, and the episode lengths and rewards are averaged over all 100 tests.
# 
# This can take 5-10 minutes. Once done, you can download the output video (2-training.mp4) to see the training progress.

# %%
# Training and testing hyperparameters
NUM_ROUNDS = 20
NUM_TRAINING_STEPS_PER_ROUND = 5000
NUM_TESTS_PER_ROUND = 100
MODEL_FILENAME_BASE = "pendulum-ppo"
VIDEO_FILENAME = "2-training.mp4"

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
            ep_len, ep_rew = test_model(env, model, f"Round {rnd}")
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


# %%
# Plot average test episode lengths and rewards for each round
fig, axs = plt.subplots(1, 2)
fig.tight_layout(pad=4.0)
axs[0].plot(avg_ep_lens)
axs[0].set_ylabel("Average episode length")
axs[0].set_xlabel("Round")
axs[1].plot(avg_ep_rews)
axs[1].set_ylabel("Average episode reward")
axs[1].set_xlabel("Round")

# %% [markdown]
# hoose a model that provided the best average rewards, load it, and call our test_model() function using that model. For example, in the plot above, we can see that the model from round 17 gave the best average test results. So, change the MODEL_FILENAME value to “pendulum-ppo_17” and run the following cell.

# %%
# Model and video settings
MODEL_FILENAME = "pendulum-ppo_17"
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

# %% [markdown]
# When you are done, don’t forget to close your environment.

# %%
env.close()


