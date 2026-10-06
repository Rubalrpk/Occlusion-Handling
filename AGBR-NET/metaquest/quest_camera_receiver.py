import socket
import struct
import cv2
import numpy as np

HOST = "127.0.0.1"
PORT = 5000

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

server.bind((HOST, PORT))
server.listen(1)

print("Waiting for Quest 3S camera...")

conn, addr = server.accept()

print("Quest connected:", addr)


def receive_exact(sock, size):
    data = b""

    while len(data) < size:
        packet = sock.recv(size - len(data))

        if not packet:
            return None

        data += packet

    return data


while True:

    size_data = receive_exact(conn, 4)

    if size_data is None:
        break

    size = struct.unpack("<I", size_data)[0]

    jpeg_data = receive_exact(conn, size)

    if jpeg_data is None:
        break

    array = np.frombuffer(
        jpeg_data,
        dtype=np.uint8
    )

    frame = cv2.imdecode(
        array,
        cv2.IMREAD_COLOR
    )

    if frame is None:
        continue

    cv2.imshow("QUEST 3S RGB CAMERA", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


conn.close()
server.close()
cv2.destroyAllWindows()