"""Advances tools for robot control, effector movement.
"""


import time
import time
from math import pi
from OperateRobot import OperateRobot
from rotation_vector_map import rotation_vector_map


class Movement():
    def __init__(self):
        self.ip = "172.31.1.25"
        self.rob = OperateRobot(self.ip)

    def move_to_pcd(self):
        self.rob.movel({"x": -0.80, "y": -0.20, "z": 0.65,
                       "rx": 1.487, "ry": 3.536, "rz": -0.669})

    def set_perpenicular_on_pcd_position(self):
        self.rob.movel({"x": -0.80, "y": -0.20, "z": 0.80,
                       "rx": 1.217, "ry": 2.882, "rz": -0.013})

    def grip_block(self, pos):
        rx, ry, rz = rotation_vector_map[pos["ang"]]
        to_block = [{"x": pos['x'], "y": pos['y'], "z": pos['z'] + 0.1, "rx": rx, "ry": ry, "rz": rz},
                    {"x": pos['x'], "y": pos['y'], "z": pos['z'],       "rx": rx, "ry": ry, "rz": rz}]

        self.rob.open_gripper()
        self.rob.movel(to_block[0])
        time.sleep(0.2)
        self.rob.movel(to_block[1])
        time.sleep(0.2)
        self.rob.close_gripper()

        # Функия для поврота последнего шарнира
    def rotate_tool_oz(self, radians):
        w1, w2, w3, w4, w5, w6 = self.rob.getj()
        print(w1, w2, w3, w4, w5, w6)
        w1 += radians
        self.rob.movej({"w1": w1, "w2": w2, "w3": w3, "w4": w4, "w5": w5, "w6": w6})
        print(self.rob.getl())

    def move_to_position(self, pos):
        rx, ry, rz = rotation_vector_map[pos["ang"]]
        self.rob.movel({"x": pos['x'], "y": pos['y'], "z": pos['z'], "rx": rx, "ry": ry, "rz": rz})

    def to_red_box(self):
        to_red_box = {"x": -0.76, "y": 0.25, "z": 0.40,
                      "rx": 1.217, "ry": 2.882, "rz": -0.013}
        self.rob.movel(to_red_box)
        time.sleep(0.2)
        self.rob.open_gripper()  # открыть

    def to_blue_box(self):
        to_blue_box = {"x": -0.94, "y": 0.25, "z": 0.40,
                       "rx": 1.217, "ry": 2.882, "rz": -0.013}
        self.rob.movel(to_blue_box)
        time.sleep(0.2)
        self.rob.open_gripper()  # открыть

    def close_gripper(self):
        self.rob.close_gripper()  
    
    def open_gripper(self):
        self.rob.open_gripper()
