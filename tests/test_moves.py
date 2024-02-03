from shutil import move
from dots3d import MatrixProcessing, Object, show_img, Object
import numpy as np
import open3d as o3d
import time
from math import pi
from OperateCamera import OperateCamera
from OperateRobot import OperateRobot
from rotation_vector_map import rotation_vector_map


class Movement():
    def __init__(self):
        self.ip = "172.31.1.25"
        self.rob = OperateRobot(self.ip)
        self.cam = OperateCamera(resolution=1)

    def get_pcd(self):
        self.rob.movel(self.cord_for_pcd)
        return self.cam.catch_frame_without_pcd()

    def move_to_pcd(self):
        self.rob.movel({"x": -0.80, "y": -0.20, "z": 0.65,
                       "rx": 1.487, "ry": 3.536, "rz": -0.669})

    def set_perpenicular_on_pcd_position(self):
        self.rob.movel({"x": -0.80, "y": -0.20, "z": 0.80,
                       "rx": 1.217, "ry": 2.882, "rz": -0.013})

    def grip_block(self, pos):
        rx, ry, rz = rotation_vector_map[pos["ang"]]
        to_block = [{"x": pos['x'], "y": pos['y'], "z": pos['z'] + 0.1,  "rx": rx, "ry": ry, "rz": rz},
                    {"x": pos['x'], "y": pos['y'], "z": pos['z'],        "rx": rx, "ry": ry, "rz": rz}]

        self.rob.open_gripper()
        self.rob.movel(to_block[0])
        time.sleep(0.2)
        self.rob.movel(to_block[1])
        time.sleep(0.2)
        self.rob.close_gripper()
    
    def move_to_position(self, pos):
        rx, ry, rz = rotation_vector_map[pos["ang"]]
        self.rob.movel({"x": pos['x'], "y": pos['y'], "z": pos['z'], "rx": rx, "ry": ry, "rz": rz})

    def to_red_box(self):
        to_red_box = {"x": -0.76, "y": 0.25, "z": 0.55,
                      "rx": 1.487, "ry": 3.536-pi/4, "rz": 0}
        self.rob.movel(to_red_box)
        time.sleep(0.2)
        self.rob.open_gripper()  # открыть

    def to_blue_box(self):
        to_blue_box = {"x": -0.94, "y": 0.25, "z": 0.55,
                       "rx": 1.487, "ry": 3.536-pi/4, "rz": 0}
        self.rob.movel(to_blue_box)
        time.sleep(0.2)
        self.rob.open_gripper()  # открыть


def test():
    movement = Movement()

    blocks = [{"x": -0.7276, "y": -0.1069, "z": 0.3807, "ang": 45},
              {"x": -0.9092, "y": -0.0915, "z": 0.3807, "ang": -45}]

    for i in range(2):
        movement.move_to_pcd()  # Move to pcd position
        time.sleep(3)  # Getting pcd
        movement.set_perpenicular_on_pcd_position()  # Set perpenicular on pcd position

        movement.grip_block(blocks[i])
        if i == 0:
            movement.to_red_box()
        else:
            movement.to_blue_box()


test()
