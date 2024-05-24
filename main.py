import gymnasium as gym
import numpy as np


class GameEnv:

    def __init__(self, env_name, EPISODES = 20, MAX_STEPS = 1000, display=False, num_env=3):
        self.env_name = env_name
        render_mode = "human" if display else ""
        self.num_env = num_env
        self.envs = gym.make_vec(env_name, num_envs=num_env, render_mode=render_mode)
        self.EPISODES = EPISODES
        self.MAX_STEPS = MAX_STEPS


    def run_test(self, show_reward=False):
        
        observations, info = self.envs.reset()
        for ep in range(self.EPISODES):

            observations, info = self.envs.reset()
            total_reward = []
            for _ in range(self.num_env):
                total_reward.append(0)

            for step in range(self.MAX_STEPS):
                actions = self.envs.action_space.sample()
                observations, rewards, terminateds, truncateds, info = self.envs.step(actions)
                for i in range(len(total_reward)):
                    total_reward[i] += rewards[i]

                
                # print(terminateds)
            
            if show_reward:
                print(total_reward)

        self.envs.close()


if __name__ == "__main__":
    env = GameEnv("LunarLander-v2", display=True, num_env=3, MAX_STEPS=150)
    env.run_test(show_reward=True)