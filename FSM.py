# Phases of the task
import numpy as np
import pinocchio as pin
MOVE_TO_OBJECT = 0
GRAB_OBJECT = 1
MOVE_TO_GOAL = 2
RELEASE_OBJECT = 3

class FSM():
    def __init__(self):
        self.state=MOVE_TO_OBJECT
    def reset(self):
        self.state=MOVE_TO_OBJECT
    def compute_action(self,observation,env):
        current_position = observation["observation"][0:3]
        object_position = observation["achieved_goal"][0:3]
        desired_position = observation['desired_goal']    
        
        if self.state == MOVE_TO_OBJECT :
            action = np.array([7,7,3]) * np.array(object_position-np.array([0,0,0.05]) - current_position)
            gripper_action = [1]  
        elif self.state  == GRAB_OBJECT:
            action = np.zeros(3)  
            gripper_action = [-1]  
        elif self.state  == MOVE_TO_GOAL :
            if np.linalg.norm(object_position - current_position) < 0.05 :
                action = 5.0 * (desired_position - current_position)
                gripper_action = [-1]  
            else:
                self.state=MOVE_TO_OBJECT
                action = 5.0 * (desired_position - current_position)
                gripper_action = [-1]
        elif self.state  == RELEASE_OBJECT:
            action = np.zeros(3)  
            gripper_action = [1]  

        full_action = np.concatenate([action, gripper_action])
        return full_action
    def state_transition(self,observation,env):
        current_position = observation["observation"][0:3]
        object_position = observation["achieved_goal"][0:3]
        desired_position = observation['desired_goal']    
        if self.state == MOVE_TO_OBJECT and np.linalg.norm(object_position - current_position) < 0.02:
            self.state = GRAB_OBJECT
        elif self.state == GRAB_OBJECT:
            self.state = MOVE_TO_GOAL
        elif self.state == MOVE_TO_GOAL and np.linalg.norm(desired_position - current_position) < 0.02:
            self.state = RELEASE_OBJECT
            
        elif self.state == RELEASE_OBJECT:
            observation, info = env.reset()        
            self.state = MOVE_TO_OBJECT