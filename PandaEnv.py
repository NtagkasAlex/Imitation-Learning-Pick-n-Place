import numpy as np
import gymnasium as gym
from panda_gym.pybullet import PyBullet
from panda_gym.envs.robots.panda import Panda
from panda_gym.envs.tasks.pick_and_place import PickAndPlace
from panda_gym.envs.core import RobotTaskEnv
import time
from Task import Mytask
import pybullet as p
class MyCustomPandaEnv(RobotTaskEnv):
    def __init__(self, render_mode="rgb_array", renderer="OpenGL", 
                 render_width=480, render_height=480, 
                 render_target_position=[0., 0., 0.], render_distance=0.4,
                 render_yaw=90, render_pitch=-40, render_roll=0,
                 reward_type="sparse", control_type="joints", shadow=False):
        sim = PyBullet(render_mode=render_mode, renderer=renderer)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)  # Disable shadows
        robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = Mytask(sim, fixed_goal=np.array([0.2,0.1,0]))
        super().__init__(robot, task, render_width=render_width, render_height=render_height, 
                         render_target_position=render_target_position, render_distance=render_distance,
                         render_yaw=render_yaw, render_pitch=render_pitch, render_roll=render_roll)
    