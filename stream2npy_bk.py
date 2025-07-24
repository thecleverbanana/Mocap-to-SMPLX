import numpy as np
import re

def rotate_x_minus_90(joints):
    rotated_joints = joints.copy()
    y = joints[..., 1].copy()
    z = joints[..., 2].copy()
    rotated_joints[..., 1] = z
    rotated_joints[..., 2] = -y
    return rotated_joints

def convert_mocap_txt_to_npy(txt_file, npy_out_file, num_joints=51):
    positions = {}

    pattern = re.compile(
        r"(\d+)\s+\(([^)]+)\)\s+\(([^)]+)\)"
    )

    with open(txt_file, 'r') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            match = pattern.match(line)
            if not match:
                print(f"[Line {line_num}] Skipped invalid line: {line}")
                continue

            try:
                joint_id = int(match.group(1)) - 1  # convert 1-based to 0-based
                pos = tuple(map(float, match.group(2).split(',')))
                positions[joint_id] = pos
            except Exception as e:
                print(f"[Line {line_num}] Error parsing: {e}")
                continue

    # Initialize array with zeros for all joints, shape (1, num_joints, 3)
    joint_array = np.zeros((1, num_joints, 3), dtype=np.float64)

    # Fill in available joint positions
    for idx, pos in positions.items():
        if 0 <= idx < num_joints:
            joint_array[0, idx] = pos
        else:
            print(f"Warning: joint index {idx} out of range (0-{num_joints-1})")

    # Apply -90 degree rotation around X-axis here
    joint_array = rotate_x_minus_90(joint_array)

    return joint_array
    # np.save(npy_out_file, joint_array)
    # print(f"Saved {npy_out_file} with shape {joint_array.shape}")

# Usage
# convert_mocap_txt_to_npy("./test_data/sample.txt", "./test_data/sample.npy", num_joints=51)
