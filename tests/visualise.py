from itertools import count
from dots3d import MatrixProcessing, Object
import numpy as np
import open3d as o3d
import cv2
import random as rng
import time
import sys
import time
from math import pi
from FrameProcessing import FrameProcessing
from OperateCamera import OperateCamera
from OperateRobot import OperateRobot
from rotation_vector_map import rotation_vector_map
from Movement import Movement


# ==========================================================================================================================
def main():
    movement = Movement()

    while True:
        
        movement.move_to_pcd()
        movement.rotate_tool_oz(np.radians(2.5))
        
        fp = FrameProcessing()
        fp.frameprocessing()
        
        obj = fp.objects[0]

        angle = ((90 - np.degrees(obj.angle)) // 5) * 5

        movement.grip_block({'x' : obj.globalx, 'y': obj.globaly, 'z': obj.globalz, 'ang': angle})
        if obj.color == [255, 0, 0]:
            movement.to_red_box()
        else:
            movement.to_blue_box()
        print(obj.color, obj.globaly, obj.globalx, obj.globalz, angle, len(obj.area), sep='    ')
            



if __name__ == "__main__":
    main()
