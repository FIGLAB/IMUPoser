import socket
import threading
from collections import deque, defaultdict
import os
from pathlib import Path
import numpy as np

import pygame
import math
from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import *
from numpy.linalg import inv
import time
from scipy.spatial.transform import Rotation as R
import time

import argparse
import select

parser = argparse.ArgumentParser()
parser.add_argument('--imus', help='the imus to use')
parser.add_argument('--save', default=True, help='if we want to save the data or not', action="store_false")
args = parser.parse_args()
_save = args.save
print(_save)

send_pred_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Internet  # UDP
OUT_UDP_PORT = 7777
OUT_UDP_IP = "127.0.0.1"

HOST = "0.0.0.0"
PORTS = [8001, 8002, 8003, 8004, 8005]
CHUNK = 1024      # buffer size for socket
BUFFER_SIZE = 50   # 300/25 secs of data = 12
# min_time_diff = 1/25.6 # seconds
min_time_diff = 1/100


KEYS = ['unix_timestamp', 'sensor_timestamp', 'accel_x', 'accel_y', 'accel_z', 'quart_x', 'quart_y', 'quart_z', 'quart_w', "roll", "pitch", "yaw"]
DATA = {}

device_ids = {
    "Left_phone": 0,
    "Left_watch": 1,
    # "Right_headphone": 2,
    "Left_headphone": 2,
    "Right_phone": 3,
    "Right_watch": 4
}

raw_acc_buffer = {id: np.zeros((BUFFER_SIZE, 3)) for id in device_ids.values()} # init with zero accel
raw_ori_buffer = {id: np.array([[0, 0, 0, 1]] * BUFFER_SIZE) for id in device_ids.values()} # init with identity rotations
calibration_quats = {id: np.array([0, 0, 0, 1]) for id in device_ids.values()}
device2bones_quats = {id: np.array([0, 0, 0, 1]) for id in device_ids.values()}
reference_times = {id: None for id in device_ids.values()}

virtual_acc = {id: np.zeros((1, 3)) for id in device_ids.values()} # init with zero accel
virtual_ori = {id: np.array([0, 0, 0, 1]) for id in device_ids.values()} # init with identity rotations

device_positions = [(-5, 0, -10.0), (-2.5, 0, -10.0), (0, 0, -10.0), (2.5, 0, -10.0), (5, 0, -10.0)]

def drawText(position, textString, size):
    font = pygame.font.SysFont("Courier", size, True)
    textSurface = font.render(textString, True, (255, 255, 255, 255), (0, 0, 0, 255))
    textData = pygame.image.tostring(textSurface, "RGBA", True)
    glRasterPos3d(*position)
    glDrawPixels(textSurface.get_width(), textSurface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, textData)

def draw_cuboid(w=2, h=2, d=0.4, colors=None):
    w = w / 2
    h = h / 2
    d = d / 2

    colors = [(0.0, 1.0, 0.0), (1.0, 0.5, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 1.0)]

    glBegin(GL_QUADS)
    glColor3f(*colors[0])

    glVertex3f(w, d, -h)
    glVertex3f(-w, d, -h)
    glVertex3f(-w, d, h)
    glVertex3f(w, d, h)

    glColor3f(*colors[1])

    glVertex3f(w, -d, h)
    glVertex3f(-w, -d, h)
    glVertex3f(-w, -d, -h)
    glVertex3f(w, -d, -h)

    glColor3f(*colors[2])

    glVertex3f(w, d, h)
    glVertex3f(-w, d, h)
    glVertex3f(-w, -d, h)
    glVertex3f(w, -d, h)

    glColor3f(*colors[3])

    glVertex3f(w, -d, -h)
    glVertex3f(-w, -d, -h)
    glVertex3f(-w, d, -h)
    glVertex3f(w, d, -h)

    glColor3f(*colors[4])

    glVertex3f(-w, d, h)
    glVertex3f(-w, d, -h)
    glVertex3f(-w, -d, -h)
    glVertex3f(-w, -d, h)

    glColor3f(*colors[5])

    glVertex3f(w, d, -h)
    glVertex3f(w, d, h)
    glVertex3f(w, -d, h)
    glVertex3f(w, -d, -h)

    glEnd()

def draw(device_id, ori, acc):
    [nx, ny, nz, w] = list(ori)
    glLoadIdentity()
    device_pos = device_positions[device_id] 

    glTranslatef(*device_pos)
    drawText((-0.7, 1.8, 0), list(device_ids.keys())[device_id], 14)
    glRotatef(2 * math.acos(w) * 180.00/math.pi, nx, nz, ny)
    draw_cuboid(1, 1, 1)

def process_data(message):
    """Receive data from socket.
    """
    message = message.strip()
    if not message:
        return
    message = message.decode('utf-8')
    if message == 'stop':
        return
    if ':' not in message:
        print(message)
        return

    try:
        device_id, raw_data_str = message.split(";")
        device_type, data_str = raw_data_str.split(':')
    except Exception as e:
        print(e, message)
        return

    data = []
    for d in data_str.strip().split(' '):
        try:
            data.append(float(d))
        except Exception as e:
            print(e)
            continue
    if len(data) != len(KEYS):
        if len(data) != len(KEYS) - 3:
            # something's missing, skip!
            print(list(np.array(data[-3:])*180/3.14))
            return

    if device_id == "left":
        device_name = device_ids[f"Left_{device_type}"]
    elif device_id == "right":
        device_name = device_ids[f"Right_{device_type}"]

    send_str = f"w{data[8]}wa{data[5]}ab{data[6]}bc{data[7]}c"

    # update the buffers
    curr_acc = np.array(data[2:5]).reshape(1, 3)
    curr_ori = np.array(data[5:9]).reshape(1, 4)
    timestamps = data[:2]

    # if device_name == 2: # headphone
    #     curr_ori[:, [1, 2]] = -1 * curr_ori[:, [2, 1]] # the headphone axes are flipped
    #     curr_ori[:, 3] = -1 * curr_ori[:, 3] # the headphone axes are flipped
    #     curr_acc *= -1;
    if device_name == 2: # headphone
        curr_euler = R.from_quat(curr_ori).as_euler("xyz").squeeze()
        fixed_euler = np.array([[curr_euler[0] * -1, curr_euler[2], curr_euler[1]]]);
        curr_ori = R.from_euler("xyz", fixed_euler).as_quat().reshape(1, 4)
        curr_acc = np.array([[curr_acc[0, 0]*-1, curr_acc[0, 2], curr_acc[0, 1]]])
        # curr_mat = R.from_quat(curr_ori).as_matrix().reshape(3,3)
        # rot_z_180 = R.from_quat([[0, 0, 1, 0]]).as_matrix().reshape(3,3)
        # rot_x_90 = R.from_quat([[0, 0, 0.707, 0.707]]).as_matrix().reshape(3,3)
        # # rot_x_90 = R.from_quat([[0, 0, 0, 1]]).as_matrix()
        # # fixed_ori = rot_x_90 @ rot_z_180 @ curr_mat
        # fixed_ori = curr_mat @ rot_z_180 @ rot_x_90
        # # fixed_ori = rot_z_180 @ curr_mat
        # # fixed_ori = curr_mat
        # curr_ori = R.from_matrix(fixed_ori).as_quat().reshape(1, 4)

    raw_acc_buffer[device_name] = np.concatenate([raw_acc_buffer[device_name][1:], curr_acc])
    raw_ori_buffer[device_name] = np.concatenate([raw_ori_buffer[device_name][1:], curr_ori])

    return send_str, device_name, list(np.array(data[2:5])), timestamps

def resizewin(width, height):
    """
    For resizing window
    """
    if height == 0:
        height = 1
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(-7, 7, -7, 7, 0, 15)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

def init():
    glShadeModel(GL_SMOOTH)
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClearDepth(1.0)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)

def sensor2global(ori, acc, device_id):
    # this function works!
    device_mean_quat = calibration_quats[device_id]

    og_mat = R.from_quat(ori).as_matrix()
    global_inertial_frame = R.from_quat(device_mean_quat).as_matrix()
    global_mat = (global_inertial_frame.T).dot(og_mat)
    global_quat = R.from_matrix(global_mat).as_quat()

    sensor_rel_acc = og_mat.dot(acc) # align acc to the sensor frame of ref
    global_acc = (global_inertial_frame.T).dot(sensor_rel_acc) # align acc to the world frame
    return global_quat, global_acc

def send_and_save_data(path_to_save, save=True):
    acc = []
    ori = []
    for _id in range(5):
        acc.append(virtual_acc[_id])
        ori.append(virtual_ori[_id][[3, 0, 1, 2]])

    a = np.array(acc)
    o = np.array(ori)

    s = ','.join(['%g' % v for v in a.flatten()]) + '#' + \
        ','.join(['%g' % v for v in o.flatten()]) + '$'

    # save the string with a unix_timestamp 
    unix_time = str(time.time())

    s_save = unix_time + "@" + s
    if save:
        with open(path_to_save, "a") as f:
            f.write(f"\n{s_save}")

    sensorBytes = bytes(s, encoding="utf8")
    send_pred_sock.sendto(sensorBytes, (OUT_UDP_IP, OUT_UDP_PORT))

if __name__ == '__main__':
    if _save:
        save_dir = Path("study_data")
        save_dir.mkdir(exist_ok=True, parents=True)

        # get the number of participants
        n_participants = len([x for x in save_dir.iterdir() if x.is_dir()])

        pid = input("Enter a participant name: ")

        path_to_save = save_dir / f"{n_participants + 1}_{pid}"
        path_to_save.mkdir(exist_ok=True, parents=True)
    else:
        path_to_save = Path("./")

    # create the sockets
    sockets = []
    for PORT in PORTS:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((HOST, PORT))
        sockets.append(sock)
        print(f"Got the {PORT} socket to bind!")


    video_flags = OPENGL | DOUBLEBUF
    pygame.init()
    screen = pygame.display.set_mode((860, 860), video_flags)
    pygame.display.set_caption("PyTeapot IMU orientation visualization")
    resizewin(860, 860)
    init()
    ticks = pygame.time.get_ticks()
    
    empty = []

    prev_timestamp = 0
    curr_timestamp = 0

    while True:
        save_s = "recording:"
        event = pygame.event.poll()

        # quit
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            break

        if (event.type == KEYDOWN and event.key == K_c):
            print("Started calibration!")
            save_s = "calibrating:"
        if (event.type == KEYUP and event.key == K_c):
            # Calc the mean quat 
            for _id in raw_ori_buffer.keys():
                mean_quat = np.mean(raw_ori_buffer[_id][-30:], axis=0) # 30 frames @20 Hz is 1.5sec
                calibration_quats[_id] = mean_quat
            save_s = "done calibrating:"
            print("Finished calibration")

        readable, writable, exceptional = select.select(sockets, empty, empty)
        for sock in readable:
            try:
                data, addr = sock.recvfrom(CHUNK)

                viz_str, device_id, accel_data, timestamps = process_data(data)

                # if it's the first timestamp of the device, save that and use it as a reference
                if reference_times[device_id] == None:
                    reference_times[device_id] = [timestamps[0], timestamps[1]]

                curr_timestamp = reference_times[device_id][0] + timestamps[1] - reference_times[device_id][1] # unix timestamp
                # print(device_id, curr_timestamp - prev_timestamp)
                # prev_timestamp = curr_timestamp

                
                o = raw_ori_buffer[device_id][-1]
                a = raw_acc_buffer[device_id][-1]

                # convert to a global inertial frame
                global_o, global_a = sensor2global(o, a, device_id)
                # global_o, global_a = o, a

                virtual_acc[device_id] = global_a.reshape(1, 3)
                virtual_ori[device_id] = global_o

                save_s += f"{' '.join([str(x) for x in timestamps])} {' '.join(list(global_a.flatten().astype(str)))} {' '.join(list(global_o.flatten().astype(str)))}"

                if _save:
                    # save to the file here
                    with open(path_to_save / f"{device_id}.csv", "a") as f:
                        f.write(f"\n{save_s}")

                if (curr_timestamp - prev_timestamp) >= min_time_diff:
                    print(f"\r{1 / (curr_timestamp - prev_timestamp)}", end='')
                    prev_timestamp = curr_timestamp
                    send_and_save_data(path_to_save=path_to_save / "sent.csv", save=_save)

                    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                    for id in virtual_ori.keys():
                        # there's a for loop here because we want to re-render the whole frame
                        _o = virtual_ori[id]
                        _a = virtual_acc[id]
                        draw(id, _o, _a)

                pygame.display.flip()
            except KeyboardInterrupt:
                print('===== close socket =====')
                os._exit(0)
            except Exception as e:
                print(e)
                pass
