from typing import Optional
from math import radians
import numpy as np
from gymnasium import spaces

from panda_gym.pybullet import PyBullet
from panda_gym.envs.core import PyBulletRobot


class Kr300R2500Ultra(PyBulletRobot):
    """KUKA KR 300 R2500 ultra robot in PyBullet.

    Args:
        sim (PyBullet): Simulation instance.
        block_gripper (bool, optional): Whether the gripper is blocked. Defaults to False.
        base_position (np.ndarray, optionnal): Position of the base base of the robot, as (x, y, z). Defaults to (0, 0, 0).
        control_type (str, optional): "ee" to control end-effector displacement or "joints" to control joint angles.
            Defaults to "ee".
    """

    def __init__(
            self,
            sim: PyBullet,
            base_position: Optional[np.ndarray] = None,
            control_type: str = "ee",
         )-> None:
        base_position = base_position if base_position is not None else np.zeros(3)
        self.control_type = control_type
        n_action = 3 if self.control_type == "ee" else 6  # control (x, y z) if "ee", else, control the 7 joints
        action_space = spaces.Box(-1.0, 1.0, shape=(n_action,), dtype=np.float32)
        super().__init__(
            sim,
            body_name="KUKA KR300 R2500 ultra",
            file_name="submodules/kuka_kr300_support/urdf/kr300r2500ultra.urdf",  # the path of the URDF file
            base_position=base_position,  # the position of the base
            action_space=action_space,
            joint_indices=np.array([0,1,2,3,4,5]),  # list of the indices, as defined in the URDF
            joint_forces=np.array([2000.0]*6),  # force applied when robot is controled (Nm)
        )
        self.neutral_joint_values = np.array([
            0.00,
            radians(-90),
            radians(90),
            0.00,
            0.00,
            0.00
            ])
        self.ee_link = 7 # tool0 link in the URDF

    def set_action(self, action: np.ndarray) -> None:
        # action = action.copy() # ensure action don't change
        # action = np.clip(action, self.action_space.low, self.action_space.high)

        # if self.control_type == "ee":
        #     ee_displacement = action[:3]
        #     target_arm_angles = self.ee_displacement_to_target_arm_angles(ee_displacement)

        self.control_joints(target_angles=action)
        
    def set_action(self, action: np.ndarray) -> None:
        action = action.copy()  # ensure action don't change
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.control_type == "ee":
            ee_displacement = action[:3]
            target_arm_angles = self.ee_displacement_to_target_arm_angles(ee_displacement)
        else:
            arm_joint_ctrl = action[:7]
            target_arm_angles = self.arm_joint_ctrl_to_target_arm_angles(arm_joint_ctrl)

        if self.block_gripper:
            target_fingers_width = 0
        else:
            fingers_ctrl = action[-1] * 0.2  # limit maximum change in position
            fingers_width = self.get_fingers_width()
            target_fingers_width = fingers_width + fingers_ctrl

        target_angles = np.concatenate((target_arm_angles, [target_fingers_width / 2, target_fingers_width / 2]))
        self.control_joints(target_angles=target_angles)

    def ee_displacement_to_target_arm_angles(self, ee_displacement: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the end-effector displacement.

        Args:
            ee_displacement (np.ndarray): End-effector displacement, as (dx, dy, dy).

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        ee_displacement = ee_displacement[:3] * 0.05  # limit maximum change in position
        # get the current position and the target position
        ee_position = self.get_ee_position()
        target_ee_position = ee_position + ee_displacement
        # Clip the height target. For some reason, it has a great impact on learning
        target_ee_position[2] = np.max((0, target_ee_position[2]))
        # compute the new joint angles
        target_arm_angles = self.inverse_kinematics(
            link=self.ee_link, position=target_ee_position, orientation=np.array([1.0, 0.0, 0.0, 0.0])
        )
        return target_arm_angles

    def get_obs(self) -> np.ndarray:
        # end-effector position and velocity
        ee_position = np.array(self.get_ee_position())
        ee_velocity = np.array(self.get_ee_velocity())

        observation = np.concatenate((ee_position, ee_velocity))
        return observation

    def reset(self) -> None:
        self.set_joint_neutral()

    def set_joint_neutral(self) -> None:
        """Set the robot to its neutral pose."""
        self.set_joint_angles(self.neutral_joint_values)

    def get_ee_position(self) -> np.ndarray:
        """Returns the position of the end-effector as (x, y, z)"""
        return self.get_link_position(self.ee_link)

    def get_ee_velocity(self) -> np.ndarray:
        """Returns the velocity of the end-effector as (vx, vy, vz)"""
        return self.get_link_velocity(self.ee_link)

if __name__ == "__main__":
    sim = PyBullet(render_mode="human")
    robot = Kr300R2500Ultra(sim)

    import time

    for _ in range(50):
        robot.set_action(np.array([0, radians(-90), radians(90), 0, 0, 0]))
        sim.step()
        time.sleep(0.1)