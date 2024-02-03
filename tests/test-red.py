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


def get_last_put(area):
    if (1 not in area):
        last_put_place = -1
    else:
        last_put_place = area.index(1)
    return last_put_place

# =========================================================================================================================
def main():

    red_area = [0, 0, 0, 0]
    take_uncomfortable = False
    movement = Movement()

    while (True):
        movement.move_to_pcd()
        movement.rotate_tool_oz(np.radians(2.5))
        fp = FrameProcessing()
        fp.frameprocessing()
        objects = fp.objects

        objects_queue = list(objects)
        queue_len = len(objects_queue)

        if (queue_len == 0):
            print("There is no blocks in the field.")
            break

        for _ in range(queue_len):
            obj = objects_queue.pop(0)

            if (obj.color == [255, 0, 0] and len(obj.area) < 550):
                is_available, angle, side = obj.get_angle(fp.mtx_cols)
                print("You should better touch edges:", side)
                # side == 1 -- с длинной стороны, 2 -- за короткую
                if (is_available or take_uncomfortable):
                    # take and put to the destination area
                    last_put_place = get_last_put(red_area)

                    if (last_put_place in [-1, 0, 1, 2]):
                        ang = (np.degrees(angle) // 5) * 5
                        movement.grip_block({'x' : obj.globalx, 'y': obj.globaly, 'z': obj.globalz, 'ang': ang})

                        if last_put_place + 2 == 1:
                            movement.to_first_red_box()
                        elif last_put_place + 2 == 2:
                            movement.to_second_red_box()
                        elif last_put_place + 2 == 3:
                            movement.to_therd_red_box()
                        else:
                            movement.to_forth_red_box()

                        red_area[last_put_place + 1] = 1

                        break
                    else: # last_put_place == 3
                        print("I found 5th red block!")
                        pass
                else:
                    objects_queue.append(obj)
            else:
                take_uncomfortable = True
    print("Finished.")

if __name__ == "__main__":
    main()
