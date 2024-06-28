from typing import Optional
import math
import numpy as np
from gymnasium import spaces

from panda_gym.pybullet import PyBullet
from panda_gym.envs.core import PyBulletRobot


class Rv7f(PyBulletRobot):
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
            displacement_scale: float = 0.15,
         )-> None:
        base_position = base_position if base_position is not None else np.zeros(3)
        self.control_type = control_type
        n_action = 3 if self.control_type == "ee" else 6  # control (x, y z) if "ee", else, control the 6 joints
        action_space = spaces.Box(-1.0, 1.0, shape=(n_action,), dtype=np.float32)
        self.dicplacement_scale = displacement_scale
        super().__init__(
            sim,
            body_name="Mitsubishi Rv7f",
            file_name="robots/mitsubishi_rv7f_description/robots/rv7f/urdf/rv7f.urdf",  # the path of the URDF file
            base_position=base_position,  # the position of the base
            action_space=action_space,
            joint_indices=np.array([0,1,2,3,4,5]),  # list of the indices, as defined in the URDF
            joint_forces=np.array([2000.0]*6),  # force applied when robot is controled (Nm)
        )
        self.neutral_joint_values = np.array([0, math.radians(-90), math.radians(90), 0, 0, 0])
        self.ee_link = 6

    def set_action(self, action: np.ndarray) -> None:
        action = action.copy()  # ensure action don't change
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.control_type == "ee":
            ee_displacement = action[:3]
            target_arm_angles = self.ee_displacement_to_target_arm_angles(ee_displacement)
        else:
            arm_joint_ctrl = action[:6]
            target_arm_angles = self.arm_joint_ctrl_to_target_arm_angles(arm_joint_ctrl)

        target_angles = target_arm_angles
        self.control_joints(target_angles=target_angles)

    def ee_displacement_to_target_arm_angles(self, ee_displacement: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the end-effector displacement.

        Args:
            ee_displacement (np.ndarray): End-effector displacement, as (dx, dy, dy).

        Returns:
            np.ndarray: Target arm angles, as the angles of the 6 arm joints.
        """
        ee_displacement = ee_displacement[:3] * self.dicplacement_scale  # limit maximum change in position
        # get the current position and the target position
        ee_position = self.get_ee_position()
        target_ee_position = ee_position + ee_displacement
        # Clip the height target. For some reason, it has a great impact on learning
        target_ee_position[2] = np.max((0, target_ee_position[2]))
        # compute the new joint angles
        target_arm_angles = self.inverse_kinematics(
            link=self.ee_link, position=target_ee_position, orientation=np.array([0.0, 0.7071068, 0.0, 0.7071068])
        )
        target_arm_angles = target_arm_angles[:6]  # remove fingers angles
        return target_arm_angles
    
    def arm_joint_ctrl_to_target_arm_angles(self, arm_joint_ctrl: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the arm joint control.

        Args:
            arm_joint_ctrl (np.ndarray): Control of the 6 joints.

        Returns:
            np.ndarray: Target arm angles, as the angles of the 6 arm joints.
        """
        arm_joint_ctrl = arm_joint_ctrl * self.dicplacement_scale  # limit maximum change in position
        # get the current position and the target position
        current_arm_joint_angles = np.array([self.get_joint_angle(joint=i) for i in range(6)])
        target_arm_angles = current_arm_joint_angles + arm_joint_ctrl
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
    robot = Rv7f(sim, control_type="joints")

    import time

    for _ in range(100):
        robot.set_action(np.array([0, math.radians(-90), math.radians(90), 0, 0, 0]))
        print(robot.get_ee_position())
        sim.step()
        time.sleep(0.1)