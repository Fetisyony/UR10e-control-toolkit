from math import degrees, sqrt, acos
import numpy as np
import cv2
import open3d as o3d
from libs.dots3d import MatrixProcessing



def main():
	PATH = r"C:\Users\Admin\Desktop\Mine\PROGRAMMING\Olympiads\NTO2021-22\tests\open-tests\1.ply"

	mt = MatrixProcessing()
	frame = mt.get_picture_from_file(PATH)

	arr = np.concatenate((np.asarray(frame.points) * 1000, np.asarray(frame.colors) * 255), axis=1)

	mt.built_mtx_ply(arr)

	mt.clean_img()

	objects = mt.create_objects()
	minobj, maxobj = mt.find_max_min_blocks(objects)

	print(mt.color_name(minobj.color))
	print(mt.color_name(maxobj.color))


if __name__ == "__main__":
	main()