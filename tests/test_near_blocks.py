from cmath import pi
import numpy as np
from math import degrees, acos
import cv2
from dots3d import MatrixProcessing


mt = MatrixProcessing()

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


img = cv2.imread(r"C:\Users\Admin\Desktop\Mine\PROGRAMMING\Olympiads\NTO2021-22\final\libraries\pic.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]
rect = cv2.minAreaRect(cnt)
box = np.int0(np.round(cv2.boxPoints(rect)))

O = list(map(int, np.round(sum(box) / 4)))
O = [O[1], O[0]]
box_list = mt.sort_clockwise([(y, x) for x, y in box], O)
box = np.array(box_list)
y_low_coord = max(box_list, key=lambda x: x[0])

ind = box_list.index(y_low_coord) # find lowest point
sorted_corners = box_list[ind:] + box_list[:ind]

a = mt.dist(sorted_corners[0], sorted_corners[1])
b = mt.dist(sorted_corners[0], sorted_corners[3])
if (a > b):
  # tilted to the left
  angle = acos((sorted_corners[1][1] - y_low_coord[1]) / a)
else:
  # tilted to the right
  angle = acos((sorted_corners[3][1] - y_low_coord[1]) / b)


def check_around(img, box, center):
	A, B, C, D = box

	A += np.int0((A - center) * 1)
	B += np.int0((B - center) * 1)
	C += np.int0((C - center) * 1)
	D += np.int0((D - center) * 1)

	img = img[min(A, B, C, D, key=lambda x: x[0])[0]:max(A, B, C, D, key=lambda x: x[0])[0], min(A, B, C, D, key=lambda x: x[1])[1]:max(A, B, C, D, key=lambda x: x[1])[1]]
	img = rotate_image(img, -degrees(angle))

	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]
	rect = cv2.minAreaRect(cnt)
	box = np.int0(np.round(cv2.boxPoints(rect)))

	O = list(map(int, np.round(sum(box) / 4)))
	O = [O[1], O[0]]
	box_list = mt.sort_clockwise([(y, x) for x, y in box], O)
	box = np.array(box_list)

	y_low_coord = min(box_list, key=lambda x: mt.dist(x, (0, 0)))

	ind = box_list.index(y_low_coord) # find lowest point
	box_list = box_list[ind-1:] + box_list[:ind-1]

	K, L, M, N = box_list
	height, width, _ = img.shape

	from PIL import Image
	Image.fromarray(img).show()

	# top and down
	top = np.count_nonzero(img[0:min(L[0], M[0]) - 5, L[1]:M[1]])
	down = np.count_nonzero(img[max(K[0], N[0]) + 5:height, K[1]:N[1]])
	print(top, down)
	if (top + down < 100):
		return 1

	# left and right
	left = np.count_nonzero(img[L[0]:K[0], 0:min(L[1], K[1]) - 5])
	right = np.count_nonzero(img[M[0]:N[0], max(M[1], N[1]) + 5:width])
	if (left + right < 100):
		return 2

	print(left, right)
	return 0

def get_angle(img, box, O):
	result = check_around(img, box, O)
	print(result)
	if (result == 1):
		return True, pi / 2 - angle
	elif (result == 2):
		if (angle >= pi / 2):
			return True, pi - angle
		else:
			return True, - angle
	return False, pi / 2 - angle

result = (get_angle(img, box, O))
print(result, degrees(result[1]))