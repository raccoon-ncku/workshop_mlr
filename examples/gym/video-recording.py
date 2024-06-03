import gymnasium as gym

###
# create a temporary variable with our env, which will use rgb_array as render mode. This mode is supported by the RecordVideo-Wrapper
tmp_env = gym.make("ALE/Pong-v5", render_mode="rgb_array")

# wrap the env in the record video
env = gym.wrappers.RecordVideo(env=tmp_env, video_folder="", name_prefix="test-video", episode_trigger=lambda x: x % 2 == 0)

# env reset for a fresh start
observation, info = env.reset()

###
# Start the recorder
env.start_video_recorder()


# AI logic
for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()

    if terminated or truncated:
        observation, info = env.reset()

####
# Don't forget to close the video recorder before the env!
env.close_video_recorder()

# Close the environment
env.close()