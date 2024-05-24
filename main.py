import gymnasium as gym


env = gym.make("LunarLander-v2", render_mode="human")



EPISODES = 20
MAX_STEPS = 1000

observation, info = env.reset()
for ep in range(EPISODES):

    observation, info = env.reset()
    for step in range(MAX_STEPS):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

env.close()