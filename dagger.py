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
from definitions import panda_env


NUM_ITS = 1000
beta_i  = 0.3
T       = 200
max_timesteps=20
def store_data(data, datasets_dir="./data"):
    # save data
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, 'data_dagger.pkl.gzip')
    f = gzip.open(data_file,'wb')
    pickle.dump(data, f)



def slow_down_loop(fps):
    sleep_time = 1.0 / fps
    time.sleep(sleep_time)

if __name__=="__main__":



    samples = {
        "state": [],
        "next_state": [],
        "reward": [],
        "action": [],
        "terminal" : [],
    }

    env = panda_env()
    desired_fps = 1000 # Adjust this value to change the speed
    fixed_goal_position = np.array([0.2, 0.0, 0.0])  # Adjust as needed for your table setup
    observation, info = env.reset()
    fsm=FSM()
    control_size=4

    running = True
    episode_rewards = []
    steps = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Model().to(device)  
    
    agent.save("dagger_models/model_0.pth")
    model_number = 0
    old_model_number =0

    for iteration in range(NUM_ITS):
        agent = Model().to(device)  
        agent.load("dagger_models/model_{}.pth".format(model_number))
        curr_beta = beta_i ** model_number 

        if model_number != old_model_number:
            print("Using model : {}".format(model_number))
            print("Beta value: {}".format(curr_beta))
            old_model_number = model_number

        episode_reward = 0
        observation, info = env.reset()
        state=env.render()
        pi = np.zeros((control_size))
        action = np.zeros_like(pi)
        timestep=0
        while True:
            
            action=fsm.compute_action(observation,env)

            observation, reward, terminated, truncated, info = env.step(pi)

            next_state=env.render()
            
            gray = np.dot(next_state[...,:3], [0.2989, 0.5870, 0.1140])
            prediction = agent(torch.from_numpy(gray[np.newaxis,np.newaxis,...].astype(np.float32)).to(device))

            pi = curr_beta * action + (1 - curr_beta) * prediction.cpu().detach().numpy().flatten()
            episode_reward += 0.1

            samples["state"].append(state)            
            samples["action"].append(np.array(action))     
            samples["next_state"].append(next_state)
            samples["reward"].append(0.1)
            samples["terminal"].append(terminated)
            
            state = next_state
            steps += 1
            timestep+=1
            # print(steps)
            if steps % T == 0:
                print('... saving data')
                store_data(samples, "./data")

                X_train, y_train = train_agent.read_data("./data", "data_dagger.pkl.gzip")
                X_train, y_train, = train_agent.preprocessing(X_train, y_train)
                train_agent.train_model(X_train, y_train, "dagger_models/model_{}.pth".format(model_number+1), num_epochs=10)
                model_number += 1
                
                break
            fsm.state_transition(observation,env)
            
            if terminated or truncated or timestep>=max_timesteps: 
                fsm.reset()

                break
        
        
                