from typing import Any, Dict

import numpy as np
from panda_gym.envs.core import Task
from panda_gym.pybullet import PyBullet
from panda_gym.utils import distance
import pybullet as p

class Mytask(Task):
    def __init__(
        self,
        sim: PyBullet,
        reward_type: str = "sparse",
        distance_threshold: float = 0.05,
        goal_xy_range: float = 0.3,
        goal_z_range: float = 0.2,
        obj_xy_range: float = 0.1,
        fixed_goal: np.ndarray = np.array([0.2, 0., 0.]),
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.object_size = 0.04
        self.goal_range_low = np.array([-goal_xy_range / 2, -goal_xy_range / 2, 0])
        self.goal_range_high = np.array([goal_xy_range / 2, goal_xy_range / 2, goal_z_range])
        self.obj_range_low = np.array([0., -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        self.fixed_goal = fixed_goal
        
        self.reset_counter = 0  # Keep track of how many times reset is called
        self.current_object_name = None  # Dynamically track the active object name (sphere or cube)
        self.object_name_sphere = "sphere_object"  # Store object name for sphere
        self.object_name_cube = "cube_object"  # Store object name for cube
        self.target_name = "target"  # Store target name
        
        # Define goals for cube and sphere
        self.goal_cube = np.array([0.2, 0.07, 0.0])
        self.goal_sphere = np.array([0.2, -0.07, 0.0])

        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        """Create the scene."""
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)

        # Create both sphere and cube objects at the start but hide them initially.
        self.sim.create_sphere(
            body_name=self.object_name_sphere,
            mass=1.0,
            radius=0.03,
            position=np.array([0.0, 0.0, -1.0]),  # Start sphere hidden
            rgba_color=np.array([0.1, 0.1, 1.0, 1.0]),
        )
        self.sim.create_box(
            body_name=self.object_name_cube,
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            position=np.array([0.0, 0.0, -1.0]),  # Start cube hidden
            rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
        )

        # Create the target object, but we will reuse the same target for both cube and sphere.
        self.sim.create_box(
            body_name=self.target_name,
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, self.object_size / 2]),  # Position the target normally
            rgba_color=np.array([0.1, 0.1, 0.1, 0.1]),
        )

    def _hide_objects(self):
        """Hide both objects by placing them under the table (out of view)."""
        # Move both sphere and cube below the table where they are hidden
        self.sim.set_base_pose(self.object_name_sphere, np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose(self.object_name_cube, np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, 0.0, 1.0]))

    def _show_object(self, object_name):
        """Move the specified object to a visible position on the table."""
        object_position = self._sample_object()  # Get a randomized position
        self.sim.set_base_pose(object_name, object_position, np.array([0.0, 0.0, 0.0, 1.0]))

    def get_obs(self) -> np.ndarray:
        """Get observation of the currently active object (sphere or cube)."""
        object_position = self.sim.get_base_position(self.current_object_name)
        object_rotation = self.sim.get_base_rotation(self.current_object_name)
        object_velocity = self.sim.get_base_velocity(self.current_object_name)
        object_angular_velocity = self.sim.get_base_angular_velocity(self.current_object_name)
        observation = np.concatenate([object_position, object_rotation, object_velocity, object_angular_velocity])
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        """Get the position of the currently active object (sphere or cube)."""
        return np.array(self.sim.get_base_position(self.current_object_name))

    def get_goal(self) -> np.ndarray:
        """Return the goal position based on the current object (cube or sphere)."""
        if self.current_object_name == self.object_name_cube:
            return self.goal_cube
        elif self.current_object_name == self.object_name_sphere:
            return self.goal_sphere
        return np.zeros(3)  # Fallback in case no object is active

    def reset(self) -> None:
        # Increment the reset counter
        self.reset_counter += 1

        # Hide all objects before showing the correct one
        self._hide_objects()

        # Show only one object (sphere or cube) based on the reset counter
        if self.reset_counter % 2 == 0:
            self.current_object_name = self.object_name_sphere  # Track the active object name
            self._show_object(self.object_name_sphere)  # Show the sphere
        else:
            self.current_object_name = self.object_name_cube  # Track the active object name
            self._show_object(self.object_name_cube)  # Show the cube

        # Set the target position based on the current object
        self.goal = self.get_goal() + np.array([-0.02, 0, self.object_size / 2])
        self.sim.set_base_pose(self.target_name, self.goal, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        """Sample a goal."""
        goal = np.array([0.0, 0.0, self.object_size / 2])  # z offset for the cube center
        noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        if self.np_random.random() < 0.3:
            noise[2] = 0.0
        goal += noise
        return goal

    def _sample_object(self) -> np.ndarray:
        """Randomize start position of object."""
        object_position = np.array([0.0, 0.0, self.object_size / 2])
        noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object_position += noise
        return object_position

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=bool)

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return -d.astype(np.float32)
