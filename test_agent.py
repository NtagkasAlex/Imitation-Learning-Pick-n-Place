import math
import numpy as np
from model  import Model
import torch
import pickle
import os
from datetime import datetime
import gzip
import json
import train_agent
import gymnasium as gym
from panda_gym.pybullet import PyBullet
from panda_gym.envs.robots.panda import Panda
from panda_gym.envs.tasks.pick_and_place import PickAndPlace
from panda_gym.envs.core import RobotTaskEnv
import time
from Task import Mytask
from PandaEnv import MyCustomPandaEnv
from FSM import FSM
import argparse
from definitions import panda_env
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Torch Device:", device)

def slow_down_loop(fps):
    sleep_time = 1.0 / fps
    time.sleep(sleep_time)

def run_episode(env, agent, rendering=True, max_timesteps=40):
    
    sum_errors = []
    step = 0
    env.reset()
    state=env.render()
    while True:
    
        gray = np.dot(state[...,:3], [0.2125, 0.7154, 0.0721])
        pred = agent(torch.from_numpy(gray[np.newaxis, np.newaxis,...]).type(torch.FloatTensor))
        a    = pred.detach().numpy().flatten()
        # env.task.show_prediction(a)
        observation, reward, terminated, truncated, info = env.step(a)
        current_position = observation["observation"][0:3]
        object_position = observation["achieved_goal"][0:3]
        desired_position = observation['desired_goal'] 

        sum_errors.append(np.linalg.norm(current_position-object_position)+np.linalg.norm(desired_position-object_position))
        state = env.render()
        step += 1
        

        if terminated or step > max_timesteps: 
            break

    return sum_errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--path", required=True, type=str, help="Path to PyTorch model")
    args = parser.parse_args()

    rendering = True                      

    n_test_episodes = 10

    # TODO: load agent
    agent = Model()
    print("Loading model {}:".format(args.path))
    agent.load(args.path)
    # agent.load("models/agent.ckpt")
    env = panda_env()
    # env.task.show_prediction(np.array([0,0.2,0,0]))
    desired_fps = 100 # Adjust this value to change the speed
    fixed_goal_position = np.array([0.2, 0.0, 0.0])  # Adjust as needed for your table setup
    observation, info = env.reset()
    control_size=4
    print("Type Something...")
    input()
    errors = []
    for i in range(n_test_episodes):
        error = run_episode(env, agent, rendering=rendering)
        errors.append(error)
    plt.figure(figsize=(10, 6))

    for episode_idx, episode_errors in enumerate(errors):
        plt.plot(episode_errors, label=f'Episode {episode_idx + 1}')

    plt.xlabel('timesteps')
    plt.ylabel('error')
    plt.title('Errors')
    plt.grid(True)
    plt.show()
    