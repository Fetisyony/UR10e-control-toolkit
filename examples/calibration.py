"""Peforms robot calibration.
Determines configuration.
"""

import cv2
import time
from math import pi
import time
from FrameProcessing import FrameProcessing
from rotation_vector_map import rotation_vector_map
from Movement import Movement


# ==========================================================================================================================
def main():
    movement = Movement()
    movement.move_to_pcd()

    start=time.time()
    fp = FrameProcessing()
    fp.frameprocessing()
    print('time: ', time.time() - start)

    move_poses = fp.calibration()
    movement.close_gripper()

    for i, pos in enumerate(move_poses):
        print(f'Moving to pose: {pos}, {i}')
        movement.move_to_position({'x': pos[1], 'y': pos[0], 'z': 0.38, 'ang': 0})
        time.sleep(1)


if __name__ == "__main__":
    main()
