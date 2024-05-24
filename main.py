import gymnasium as gym
import numpy as np
import neat
import neat.config
import os
import pickle


class GameMutipleEnv:

    def __init__(self, env_name, EPISODES = 20, MAX_STEPS = 1000, display=False, num_env=3):
        self.env_name = env_name
        render_mode = "human" if display else ""
        self.num_env = num_env
        self.env = gym.make_vec(env_name, render_mode=render_mode, num_envs=num_env)
        self.EPISODES = EPISODES
        self.MAX_STEPS = MAX_STEPS


    def run_test(self, show_reward=False):
        
        observation, info = self.env.reset()

        for ep in range(self.EPISODES):

            observations, info = self.env.reset()
            total_reward = 0

            for step in range(self.MAX_STEPS):
                action = self.env.action_space.sample()
                observation, rewards, terminated, truncated, info = self.env.step(action)
                total_reward += rewards

                
                if terminated or truncated:
                    break
            
            if show_reward:
                print(total_reward)

        self.env.close()

    def train_ai(self, genomes, config):
        nets = []
        for genome in genomes:
            nets.append(neat.nn.FeedForwardNetwork.create(genome[1], config))

        observation, info = self.env.reset()
        total_reward = []
        for _ in range(self.num_env):
            total_reward.append(0)

        for step in range(self.MAX_STEPS):
            actions = []
            for i in range(len(nets)):
                output = nets[i].activate(tuple(observation[i]))
                actions.append(output.index(max(output)))
            
            observation, rewards, terminated, truncated, info = self.env.step(actions)
            for i in range(self.num_env):
                total_reward[i] += rewards[i]
                
            # if terminated or truncated:
            #     break

        for i in range(len(genomes)):
            genomes[i][1].fitness = total_reward[i]

class GameSingleEnv:
    def __init__(self, env_name, EPISODES = 20, MAX_STEPS = 1000, display=False):
        self.env_name = env_name
        render_mode = "human" if display else ""

        self.env = gym.make(env_name, render_mode=render_mode)
        self.EPISODES = EPISODES
        self.MAX_STEPS = MAX_STEPS

    def run_test(self, show_reward=False):
        
        observation, info = self.env.reset()

        for ep in range(self.EPISODES):

            observations, info = self.env.reset()
            total_reward = 0

            for step in range(self.MAX_STEPS):
                action = self.env.action_space.sample()
                observation, rewards, terminated, truncated, info = self.env.step(action)
                total_reward += rewards

                
                if terminated or truncated:
                    break
            
            if show_reward:
                print(total_reward)

        self.env.close()

    def train_ai(self, genome, config):
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        observation, info = self.env.reset()
        total_reward = 0

        for step in range(self.MAX_STEPS):

            output = net.activate(tuple(observation))
            action = output.index(max(output))
            
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
                
            if terminated or truncated:
                break

        genome.fitness = total_reward

def eval_genomes(genomes, config):
    # Create Environments
    # NUM_ENV = 5
    # env = GameMutipleEnv("LunarLander-v2", display=True, MAX_STEPS=150, num_env=NUM_ENV)
    env = GameSingleEnv("LunarLander-v2", display=True, MAX_STEPS=150)

    # Run genome
    for i, (genome_id, genome) in enumerate(genomes):
        env.train_ai(genome, config)

    # print(genomes[0][1])
    # genome_list = list(map(lambda g: g[1], genomes))
    # for i in range(0, len(genomes), NUM_ENV):
    #     env.train_ai(genomes[i:i+NUM_ENV], config)
    # print(genomes[0])

def run_neat(config):
    p = neat.Checkpointer.restore_checkpoint("neat-checkpoint-11") # load checkpoint
    # p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    winner = p.run(eval_genomes, 3)

    # save the best genome
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)


if __name__ == "__main__":
    # Test
    # env = GameEnv("LunarLander-v2", display=False, MAX_STEPS=500)
    # env.run_test(show_reward=True)


    # NEAT
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    run_neat(config)