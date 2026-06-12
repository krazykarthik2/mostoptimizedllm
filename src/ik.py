import numpy as np
import ikpy.chain
import ikpy.utils.plot as plot_utils

class RobotIK:
    def __init__(self, urdf_path):
        self.chain = ikpy.chain.Chain.from_urdf_file(urdf_path)
        # 9 links: ['Base link', 'joint_a1', 'joint_a2', 'joint_a3', 'joint_a4', 'joint_a5', 'joint_a6', 'gripper_base_joint', 'gripper_left_joint']
        # Active: joints a1 to a6 (indices 1 to 6)
        self.chain.active_links_mask = [False] + [True]*6 + [False]*(len(self.chain.links)-7)
        
    def solve_ik(self, target_position, target_orientation=None):
        """
        target_position: [x, y, z]
        target_orientation: Optional 3x3 rotation matrix or RPY
        """
        joint_angles = self.chain.inverse_kinematics(
            target_position, 
            target_orientation=target_orientation, 
            orientation_mode="all" if target_orientation is not None else None
        )
        return joint_angles

    def forward_kinematics(self, joint_angles):
        return self.chain.forward_kinematics(joint_angles)

if __name__ == "__main__":
    # ik = RobotIK("urdf/robot.urdf")
    # target = [0.2, 0.1, 0.3]
    # angles = ik.solve_ik(target)
    # print(f"Joint angles: {angles}")
    pass
