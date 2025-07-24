# main.py
import socket
import pickle
import numpy as np

HOST = '127.0.0.1'
PORT = 9999

def recv_exact(sock, size):
    """Helper to receive exactly `size` bytes"""
    data = b''
    while len(data) < size:
        more = sock.recv(size - len(data))
        if not more:
            raise EOFError("Socket closed")
        data += more
    return data

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    print(f"[Main] Connecting to {HOST}:{PORT}...")
    s.bind((HOST, PORT))
    s.listen(1)
    conn, addr = s.accept()
    print(f"[Main] Connected by {addr}")

    while True:
        try:
            size_data = recv_exact(conn, 4)
            size = int.from_bytes(size_data, 'big')
            payload = recv_exact(conn, size)
            frame = pickle.loads(payload)
            print("Frame Start:")
            for rigid_id, pos in frame:
                print(f"  ID: {rigid_id}, Pos: {np.array(pos)}")
            print("Frame End.")

        except Exception as e:
            print("[Main] Error:", e)
            break
