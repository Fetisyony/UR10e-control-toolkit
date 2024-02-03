from math import degrees, sqrt, acos
import numpy as np
import cv2
import open3d as o3d
from dots3d import MatrixProcessing



SHOW_IMG_ACCESS = 1
SAVE_IMG_ACCESS = 1

def show_img(mtx):
	if SHOW_IMG_ACCESS == 0: return
	from PIL import Image
	Image.fromarray(np.array(mtx, dtype=np.uint8)).show()
def save_img(mtx, name=1):
	if SAVE_IMG_ACCESS == 0: return
	import cv2
	cv2.imwrite(rf"{name}.png", cv2.cvtColor(np.array(mtx, dtype=np.uint8), cv2.COLOR_BGR2RGB))

def main():
	# ============================ get_picture ===========================================
	mt = MatrixProcessing()
	# cam = OperateCamera()
	# frame = cam.catch_frame()
	frame = mt.get_pic_from_file(r"C:\Users\Admin\Desktop\Mine\PROGRAMMING\Olympiads\NTO2021-22\final\libraries\6.ply")
	arr = np.concatenate((np.asarray(frame.points) * 1000, np.asarray(frame.colors) * 255), axis=1)
	from time import time
	m = time()
	mt.build_mt(arr)
	mt.mt_color = [[[i for i in x] for x in line] for line in mt.mt_color]
	mt.trinarise_mtx()
	
	mt.clean_img(min_area=50)
	mt.clean_img(min_area=50)
	# ======================= get_picture and process ====================================

	# 573.2145804676754 -- 2 level
	# 600 -- 1 level
	
	main_set = set()
	areas = []
	img = [[[i for i in x] for x in line] for line in mt.mt_color]

	for y in range(mt.HEIGHT):
		for x in range(mt.WIDTH):
			if (img[y][x] != [0, 0, 0] and ((y, x) not in main_set)):
				visited = mt.get_area((y, x))
				main_set = main_set | visited
				areas.append(list(visited))
	red = 0
	blue = 0
	for area in areas:
		if (len(area) < 140):
			continue
		color = mt.color_name(mt.most_common_color(area))
		if (color == "RED"):
			red += 1
		elif (color == "BLUE"):
			blue += 1
	print(blue, red)
	print(time() - m)

if __name__ == "__main__":
	main()