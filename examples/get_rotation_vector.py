import time
import numpy as np
import os
from OperateCamera import OperateCamera
from OperateRobot import OperateRobot


rob = OperateRobot("172.31.1.25")
cam = OperateCamera()

# Едем  изначальную позицию для фотографироания, под 45 градусо к столу
print('Frame capture pos')
rob.movel({"x": -0.80, "y": -0.20, "z": 0.80,
          "rx": 1.487, "ry": 3.536, "rz": -0.669})
time.sleep(5)

# Поворачиаемся перпендикулярно плоскости стола
print('Move to st pos...')
rob.movel({"x": -0.80, "y": -0.20, "z": 0.80,
          "rx": 1.217, "ry": 2.882, "rz": -0.013})
time.sleep(5)

# Функия для поврота последнего шарнира
def rotate_tool_oz(radians):
    w1, w2, w3, w4, w5, w6 = rob.getj()
    print(w1, w2, w3, w4, w5, w6)
    w6 = radians
    rob.movej({"w1": w1, "w2": w2, "w3": w3, "w4": w4, "w5": w5, "w6": w6})


# Поворачиваемся на каждые 5 градусов и сохраняем значения rx, ry, rz
positions = []
_, _, _, _, _, w = rob.getj()
for i in range(-180, 181, 5):
    angle_now = w + np.radians(i)
    rotate_tool_oz(angle_now)
    _, _, _, rx, ry, rz = rob.getl()
    positions.append([i, rx, ry, rz])
    print(i, rx, ry, rz)
path = os.path.join(os.getcwd(), 'output', 'position_vector.npy')
np.save(path, np.array(positions))


rob.close()
cam.stop()
