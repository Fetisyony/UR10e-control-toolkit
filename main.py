"""Designed for solving the problem of
recognition and sorting objects into groups by color.
"""


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




def bulks_coords(blocks):
    """Evaluates ways for removing bulks of blocks.
    :blocks: list of recognized blocks
    :returns: list of coords in correct sequence.
    """
    move_poses = []
    for i, obj in enumerate(blocks):
        move_poses.append((obj.globaly + 0.07, obj.globalx))
        move_poses.append((obj.globaly - 0.07, obj.globalx))
        move_poses.append((obj.globaly, obj.globalx + 0.07))
        move_poses.append((obj.globaly, obj.globalx - 0.07))
    return move_poses


def main():
    movement = Movement()

    # moves effector to default position
    movement.move_to_pcd()
    movement.rotate_tool_oz(np.radians(2.5))

    # run blocks recognition
    fp = FrameProcessing()
    fp.frameprocessing()

    # leave one level of blocks
    move_poses = bulks_coords(fp.objects)

    # elimitaning bulks of blocks
    movement.close_gripper()
    for pos in move_poses:
        movement.move_to_position({'x': pos[1], 'y': pos[0], 'z': 0.38, 'ang': 0})


    while 1:
        # moves effector to default position
        movement.move_to_pcd()
        movement.rotate_tool_oz(np.radians(2.5))

        fp = FrameProcessing()
        fp.frameprocessing()

        if (not fp.objects):
            break
        
        # choosing and object
        obj = fp.objects[0]

        angle = ((90 - np.degrees(obj.angle)) // 5) * 5

        # distribution into boxes
        movement.grip_block({'x' : obj.globalx, 'y': obj.globaly, 'z': obj.globalz, 'ang': angle})
        if obj.color == [255, 0, 0]:
            movement.to_red_box()
        else:
            movement.to_blue_box()


if __name__ == "__main__":
    main()
