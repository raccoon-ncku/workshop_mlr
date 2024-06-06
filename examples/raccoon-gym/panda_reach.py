from panda_gym.envs.panda_tasks import PandaReachEnv

if __name__ == "__main__":
    env = PandaReachEnv(render_mode="human", renderer="OpenGL")

    observation, info = env.reset()

    for _ in range(1000):
        action = env.action_space.sample() # random action
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()