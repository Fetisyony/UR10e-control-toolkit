"""Removes bulks of blocks.
"""


import cv2
import time
from math import pi
import time
from FrameProcessing import FrameProcessing
from rotation_vector_map import rotation_vector_map
from Movement import Movement
import numpy as np


def way_to_remove_top_blocks(blocks):
    move_poses = []
    for i, obj in enumerate(blocks):
        move_poses.append((obj.globaly + 0.07, obj.globalx + 0.01))
        move_poses.append((obj.globaly - 0.07, obj.globalx - 0.01))
        print('object'+str(i),obj.color, obj.globaly, obj.globalx, obj.globalz, obj.angle, len(obj.area), sep='    ')
    return move_poses

# ==========================================================================================================================
def main():

    movement = Movement()
    movement.move_to_pcd()
    movement.rotate_tool_oz(np.radians(2.5))

    start=time.time()
    fp = FrameProcessing()
    fp.frameprocessing()
    print('time: ', time.time()-start)

    move_poses = way_to_remove_top_blocks(fp.objects)

    movement.close_gripper()
    for pos in move_poses:
        movement.move_to_position({'x': pos[1], 'y': pos[0], 'z': 0.38, 'ang': 0})


if __name__ == "__main__":
    main()
