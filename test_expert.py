import numpy as np
import gymnasium as gym
from panda_gym.pybullet import PyBullet
from panda_gym.envs.robots.panda import Panda
from panda_gym.envs.tasks.pick_and_place import PickAndPlace
from panda_gym.envs.core import RobotTaskEnv
import time
from Task import Mytask
from PandaEnv import MyCustomPandaEnv
from FSM import FSM
# Usage example
import imageio
from definitions import panda_env

env = panda_env()

# Define a function to slow down the loop
def slow_down_loop(fps):
    sleep_time = 1.0 / fps
    time.sleep(sleep_time)

desired_fps = 1000  # Adjust this value to change the speed
fixed_goal_position = np.array([0.2, 0.0, 0.0])  # Adjust as needed for your table setup
observation, info = env.reset()
fsm=FSM()

# Slow down the loop to make the movements and rendering slower
for _ in range(1000):
    action = fsm.compute_action(observation,env)

    observation, reward, terminated, truncated, info = env.step(action)
    current_position = observation["observation"][0:3]
    object_position = observation["achieved_goal"][0:3]
    desired_position = observation["desired_goal"][0:3]
    # Slow down the loop to control the simulation speed
    slow_down_loop(desired_fps)
    
    image = env.render()
    gray = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

    # Increment the step counter
    # print(image.shape)
    # Phase transitions with increased thresholds
    
    fsm.state_transition(observation,env)
    # Check if maximum steps per episode is reached to avoid premature truncation
    # print(truncasignalsted)
    if terminated  or truncated:
        observation, info = env.reset()
        fsm.reset()
    
env.close()
imageio.imwrite('panda_robot_image.png', image)
# imageio.imwrite('grayscale_image.png', gray)