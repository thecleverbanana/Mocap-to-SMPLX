import socket
import pickle
import numpy as np

# === Settings ===
HOST = '127.0.0.1'
PORT = 9999
NUM_JOINTS = 51
# SAVE_PATH = './latest_frame.npy'
SAVE_PATH = './all_frame.npy'

def recv_exact(sock, size):
    """Receive exactly `size` bytes from socket"""
    data = b''
    while len(data) < size:
        more = sock.recv(size - len(data))
        if not more:
            raise EOFError("Socket closed")
        data += more
    return data

def rotate_x_minus_90(joints):
    """Rotate joints -90 degrees around the X-axis (YZ swap + Z negated)"""
    rotated_joints = joints.copy()
    y = joints[..., 1].copy()
    z = joints[..., 2].copy()
    rotated_joints[..., 1] = z
    rotated_joints[..., 2] = -y
    return rotated_joints

def convert_frame_to_npy(frame, num_joints=51):
    """Convert list of (id, pos) to npy of shape (1, num_joints, 3)"""
    joint_array = np.zeros((1, num_joints, 3), dtype=np.float64)
    for rigid_id, pos in frame:
        index = rigid_id - 196609
        if 0 <= index < num_joints:
            joint_array[0, index] = pos
    return rotate_x_minus_90(joint_array)

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print(f"[stream2npy] Waiting for connection on {HOST}:{PORT}...")
        s.bind((HOST, PORT))
        s.listen(1)
        conn, addr = s.accept()
        print(f"[stream2npy] Connected by {addr}")

        all_frames = []

        while True:
            try:
                size_data = recv_exact(conn, 4)
                size = int.from_bytes(size_data, 'big')
                payload = recv_exact(conn, size)
                frame = pickle.loads(payload)  # list of (id, pos)

                npy_data = convert_frame_to_npy(frame, num_joints=NUM_JOINTS)

                # np.save(SAVE_PATH, npy_data)

                all_frames.append(npy_data)
                stacked = np.vstack(all_frames)
                np.save(SAVE_PATH, stacked)

            except Exception as e:
                print("[stream2npy] Error:", e)
                break

if __name__ == "__main__":
    main()
