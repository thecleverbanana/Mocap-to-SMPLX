import numpy as np
import torch.nn as nn
import torch

OPTITRACK_SKEL = [
    'Hips',
    'SpineLow','SpineHight','Neck','Head',
    'LeftShoulder','LeftArm','LeftForeArm','LeftHand',
    'RightShoulder','RightArm','RightForeArm','RightHand',
    'LeftUpLeg','LeftLeg','LeftFoot','LeftFootToeBase',
    'RightUpLeg','RightLeg','RightFoot','RightFootToeBase',
        'L_Thumb1', 'L_Thumb2', 'L_Thumb3',
        'L_Index1', 'L_Index2', 'L_Index3',
        'L_Middle1', 'L_Middle2', 'L_Middle3',
        'L_Ring1', 'L_Ring2', 'L_Ring3',
        'L_Pinky1', 'L_Pinky2', 'L_Pinky3',
        'R_Thumb1', 'R_Thumb2', 'R_Thumb3',
        'R_Index1', 'R_Index2', 'R_Index3',
        'R_Middle1', 'R_Middle2', 'R_Middle3',
        'R_Ring1', 'R_Ring2', 'R_Ring3',
        'R_Pinky1', 'R_Pinky2', 'R_Pinky3'
]

SELECTED_JOINTS=np.concatenate(
    [range(0,21)]
)

OPTITRACK_BODY=np.concatenate(
    [range(0,21)]
)

OPTITRACK_HAND=np.concatenate(
    [range(22,51)]
)


OPTITRACK_TO_SMPLX = np.array([
    0,              # pelvis
    3,6,12,15, # SpineLow, SpineHight, Neck, Head
    13,16,18,20,     # LeftShoulder, LeftArm, LeftForeArm, LeftHand
    14,17,19,21,     # RightShoulder, RightArm, RightForeArm, RightHand
    1,4,7,10,    # LeftUpLeg, LeftLeg, LeftFoot, LeftFootToeBase
    2,5,8,11    # RightUpLeg, RightLeg, RightFoot, RightFootToeBase
        # 37,38,39,
        # 27,28,29,
        # 30,31,32,
        # 36,37,38,
        # 33,34,35,
        # 54,55,56,
        # 40,41,42,
        # 43,44,45,
        # 49,50,51,
        # 46,47,48
])


class JointMapper(nn.Module):
    def __init__(self, joint_maps=None):
        super(JointMapper, self).__init__()
        if joint_maps is None:
            self.joint_maps = joint_maps
        else:
            self.register_buffer('joint_maps',
                                 torch.tensor(joint_maps, dtype=torch.long))

    def forward(self, joints, **kwargs):
        if self.joint_maps is None:
            return joints
        else:
            return torch.index_select(joints, 1, self.joint_maps)

# class JointMapper(nn.Module):
#     def __init__(self, joint_maps=None, joint_names=None):
#         super(JointMapper, self).__init__()
#         if joint_maps is None:
#             self.joint_maps = None
#         else:
#             self.register_buffer('joint_maps',
#                                  torch.tensor(joint_maps, dtype=torch.long))
#         self.joint_names = joint_names

#     def forward(self, joints, **kwargs):
#         if self.joint_maps is None:
#             return joints
#         else:
#             return torch.index_select(joints, 1, self.joint_maps)
    
#     def print_mapped_joints(self):
#         if self.joint_names is None or self.joint_maps is None:
#             print("Joint names or maps not provided.")
#             return
#         print("Mapped joints:")
#         for i, idx in enumerate(self.joint_maps):
#             print(f"{i}: {self.joint_names[idx]}")

        
if __name__=='__main__':
    print(SELECTED_JOINTS)
    print(len(SELECTED_JOINTS))