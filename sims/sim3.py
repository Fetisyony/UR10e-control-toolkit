"""Определяет в какой координатной четверти плоскости 𝑋𝑌
находится новый объект. Гарантируется что данный объект находится
полностью в этой четверти.

Входные данные: через файл input.ply управляющей программе передается облако точек.
Ожидаемый результат: В результате работы программы в консоль выведено целое число — номер координатной четверти.
"""


import cv2
import numpy as np
from math import degrees, sqrt, acos
import open3d as o3d
from time import sleep, time



class Tools:
	def __init__(self) -> None:
		self.red_low_1 = [0, 0.3, 0.35]
		self.red_high_1 = [26, 1, 1]
		self.red_low_2 = [340,0.4, 0.4]
		self.red_high_2 = [360, 1, 1]

		self.blue_low = [185, 0.3, 0.2]
		self.blue_high = [245, 1, 1]

	def get_pic_from_file(self, path):
		pcd = o3d.io.read_point_cloud(path)

		return pcd

	def dist(self, p1, p2):
		y1, x1 = p1
		y2, x2 = p2
		return sqrt((y2 - y1) * (y2 - y1) + (x2 - x1) * (x2 - x1))

	def get_next_pnts(self, curPnt, short_set=True):
		nextPnts = []

		if curPnt[0] != 0:
			nextPnts.append((curPnt[0] - 1, curPnt[1]))
		if curPnt[1] != 0:
			nextPnts.append((curPnt[0], curPnt[1] - 1))
		if curPnt[0] != self.HEIGHT-1:
			nextPnts.append((curPnt[0] + 1, curPnt[1]))
		if curPnt[1] != self.WIDTH-1:
			nextPnts.append((curPnt[0], curPnt[1] + 1))
		if short_set:
			return nextPnts

		# will return 8 points around curPnt
		if curPnt[0] != 0 and curPnt[1] != 0:
			nextPnts.append((curPnt[0] - 1, curPnt[1] - 1))
		if curPnt[0] != 0 and curPnt[1] != self.WIDTH-1:
			nextPnts.append((curPnt[0] - 1, curPnt[1] + 1))
		if curPnt[0] != self.HEIGHT-1 and curPnt[1] != 0:
			nextPnts.append((curPnt[0] + 1, curPnt[1] - 1))
		if curPnt[0] != self.HEIGHT-1 and curPnt[1] != self.WIDTH-1:
			nextPnts.append((curPnt[0] + 1, curPnt[1] + 1))
		return nextPnts

	def sort_clockwise(self, points, center=[0, 0]):
		""" points: [[y, x], [y, x], ..]
		Just sorts given points clockwise.
		"""
		vectors = [[vec[0] - center[0], vec[1] - center[1]] for vec in points]
		axe_v = [0, 1000]

		angles_dict = {}
		for vec in vectors:
			if vec[0] <= 0:
				k1 = -1
			else:
				k1 = 1
			cosinus = (vec[1] * axe_v[1] + vec[0] * axe_v[0]) / (sqrt(vec[1] * vec[1] + vec[0] * vec[0]) * sqrt(axe_v[1] * axe_v[1] + axe_v[0] * axe_v[0]))

			if cosinus < -1:
				cosinus = -1
			if cosinus > 1:
				cosinus = 1
			if k1 * acos(cosinus) not in angles_dict:
				angles_dict[k1 * acos(cosinus)] = []
			angles_dict[k1 * acos(cosinus)].append(vec)

		sorted_angles = sorted(angles_dict)
		clockwise = [[point[0] + center[0], point[1] + center[1]] for angle in sorted_angles for point in angles_dict[angle]]
		return clockwise

	def rgb2hsv(self, pixel):
		"""Don't use this method outside the function.
		"""
		r = pixel[0] / 255
		g = pixel[1] / 255
		b = pixel[2] / 255
		maxC = max([r, g, b])
		minC = min([r, g, b])

		diff = maxC - minC
		if diff == 0:
			hue = 0
		elif maxC == r:
			hue = 60 * ((g - b) / diff % 6)
		elif maxC == g:
			hue = 60 * ((b - r) / diff + 2)
		elif maxC == b:
			hue = 60 * ((r - g) / diff + 4)

		if maxC == 0:
			sat = 0
		else:
			sat = diff / maxC

		return [hue, sat, maxC]

	def is_red(self, pixel):
		"""Takes RGB pixel as input.
		"""
		pixel = self.rgb2hsv(pixel)
		for i in range(3):
			if not (((self.red_low_1[i] <= pixel[i]) and (pixel[i] <= self.red_high_1[i])) or ((self.red_low_2[i] <= pixel[i]) and (pixel[i] <= self.red_high_2[i]))):
				return False
		return True

	def is_blue(self, pixel):
		"""Takes RGB pixel as input.
		"""
		pixel = self.rgb2hsv(pixel)
		for i in range(3):
			if not ((self.blue_low[i] <= pixel[i]) and (pixel[i] <= self.blue_high[i])):
				return False
		return True

	def most_common_color(self, area):
		red = 0
		blue = 0
		for (y, x) in area:
			color = self.color_name(self.mt_color[y][x])
			if (color == [255, 0, 0]):
				red += 1
			elif (color == [0, 0, 255]):
				blue += 1
		if (red > blue):
			return [255, 0, 0]
		else:
			return [0, 0, 255]

	def color_name(self, pixel):
		if (pixel == [255, 0, 0]):
			return "RED"
		elif (pixel == [0, 0, 255]):
			return "BLUE"
		else:
			return "UNKNOWN COLOR"


class Object(Tools):
	def __init__(self, points, heights, img, color=[0, 0, 0]) -> None:
		self.area = points #Point [(y, x), (y, x), ....]
		if (type(heights) is dict):
			self.heights = dict(heights)
		else:
			self.heights = {}
			for i, point in enumerate(points):
				self.heights[point] = heights[i]
		self.bare_img = [[[i for i in x] for x in line] for line in img]
		self.color = color

		self.contourxy = self.get_contourxy()
		self.contourz  = self.get_contourz()

		#self.centery, self.centerx, self.centerz = self.get_center()

		self.normal = self.get_normal()

		img = cv2.cvtColor(np.array(img, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
		cvcontours = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
		cnt = sorted(cvcontours, key=cv2.contourArea, reverse=True)[0]
		rect = cv2.minAreaRect(cnt)
		self.box = np.int0(cv2.boxPoints(rect))

		center = np.round(sum(self.box) / 4)
		self.centerx, self.centery = map(int, center)
	
		self.centerz = sum(self.heights.values()) / len(heights) # medium between all the heights of the surface

	def get_normal(self):
		xs = [i[1] for i in self.area]
		ys = [i[0] for i in self.area]
		zs = [-self.heights[i] for i in self.area]

		# do fit
		tmp_A = []
		tmp_b = []
		for i in range(len(xs)):
			tmp_A.append([xs[i], ys[i], 1])
			tmp_b.append(zs[i])
		b = np.matrix(tmp_b).T
		A = np.matrix(tmp_A)
		fit = (A.T * A).I * A.T * b
		errors = b - A * fit
		residual = np.linalg.norm(errors)
		A, B, C = -float(fit[0]), -float(fit[1]), 1
		A, B, C = A/np.sqrt(A*A+B*B+C*C), B/np.sqrt(A*A+B*B+C*C), C/np.sqrt(A*A+B*B+C*C)
		normal = (A, B, C)
		return normal

	def get_contourxy(self):
		points = set(self.area)
		contourxy = []
		for point in points:
			close_points = []
			for i in range(-1, 2):
				for j in range(-1, 2):
					if i * i + j * j == 1: close_points.append((point[0] + i, point[1] + j))
			x = 0
			for close_point in close_points:
				if close_point in points:
					x += 1
			if x != 4:
				contourxy.append(point)
		return contourxy

	def get_contourz(self):
		contourz = []
		for point in self.contourxy:
			contourz.append(self.heights[point])
		return contourz

	def get_center(self):
		centery = centerx = 0
		for y, x in self.contourxy:
			centery += y
			centerx += x
		centery = int(np.round(centery / len(self.contourxy)))
		centerx = int(np.round(centerx / len(self.contourxy)))
		return centery, centerx, self.heights[(centery, centerx)]

	def visualize(self):
		import open3d as o3d
		x, y, z = [], [], []
		cr, cg, cb = [], [], []

		x.append(self.centerx)
		y.append(self.centery)
		z.append(self.centerz)
		cr.append(0)
		cg.append(0)
		cb.append(255)

		for i ,j in self.area:
			x.append(j)
			y.append(i)
			z.append(self.heights[(i, j)])
			if((i, j) not in self.contourxy):
				cr.append(self.color[0])
				cg.append(self.color[1])
				cb.append(self.color[2])
			else:
				cr.append(0)
				cg.append(0)
				cb.append(0)

		for i in range(15):
			x.append(self.centerx + self.normal[0] * i)
			y.append(self.centery + self.normal[1] * i)
			z.append(self.centerz - self.normal[2] * i)
			cr.append(0)
			cg.append(255)
			cb.append(0)

		for i in range(-100, 100):
			x.append(self.centerx + i)
			y.append(self.centery)
			z.append(self.centerz + 50)
			cr.append(0)
			cg.append(0)
			cb.append(0)

		for i in range(-100, 100):
			x.append(self.centerx)
			y.append(self.centery + i)
			z.append(self.centerz + 50)
			cr.append(0)
			cg.append(0)
			cb.append(0)

		points = np.vstack((x, y, z)).transpose()
		colors = np.vstack((cr, cg, cb)).transpose()

		pcd = o3d.geometry.PointCloud()
		pcd.points = o3d.utility.Vector3dVector(points)
		pcd.colors = o3d.utility.Vector3dVector(colors/255)
		o3d.visualization.draw_geometries([pcd])

	def view_img(self):
		test_img = np.array([[list(i) for i in line] for line in self.bare_img])
		cv2.drawContours(test_img, [self.box], 0, (0, 255, 0), 1)
		return test_img


class HeightMatrixProc:
	"""Functions for heights matrix processing.
	"""
	def get_area_h(self, pixel):
		"""pixel = (y, x)
		Returns area near the given point which differ from (y, x) coordinate not more than COEFF.
		"""

		COEFF = 8
		height = self.mt_height[pixel[0]][pixel[1]]

		visited = set([pixel])
		queue = set([pixel])
		heights_area = [pixel]

		while queue:
			nextPnts = self.get_next_pnts(queue.pop())
			for next_point in nextPnts:
				if (next_point not in visited):
					visited.add(next_point)
					# we can also do another one thing here: check if inverse of matcher is not self.mt_color[next_point[0]][next_point[1]]
					if abs(self.mt_height[next_point[0]][next_point[1]] - height) < COEFF and [255, 0, 0] == self.mt_color[next_point[0]][next_point[1]]:
						queue.add(next_point)
						heights_area.append(next_point)


class RobotControl:
	def get_pic_from_robot():
		cam = OperateCamera()

		return cam.catch_frame()


class MatrixProcessing(Tools):
	def __init__(self, path=None, path_to_img=None, name=1):
		super().__init__()

		if (path):
			self.PATH = path
		if (path_to_img):
			self.PATH_TO_IMG = path_to_img

		self.mt_color = [[[255, 255, 255]]] # begin with this for convenience
		self.mt_height = [[0]]

		self.HEIGHT, self.WIDTH = None, None
		# top left and low right corners
		# we don't care about z
		self.tl_corner = [0, 0] # xl, yu | indexes!!
		self.lr_corner = [0, 0] # xr, yl

		self.objects = []

	def build_mt(self, matrix):
		def do_expend(number, axis):
			result = [0, 0] # low not ok, high not ok
			if (number < self.tl_corner[axis]):
				result[0] = 1
			elif (number > self.lr_corner[axis]):
				result[1] = 1
			return result
		def add_colomns(number_colomns, where):
			if (where):
				# add after
				for i in range(len(self.mt_color)):
					self.mt_color[i].extend([[0, 0, 0]] * number_colomns)
					self.mt_height[i].extend([0] * number_colomns)
			else:
				# add before
				for _ in range(number_colomns):
					for i in range(len(self.mt_color)):
						self.mt_color[i].insert(0, [0, 0, 0])
						self.mt_height[i].insert(0, 0)
		def add_lines(number_lines, where):
			l = len(self.mt_color[0])
			if (where):
				# add after
				for _ in range(number_lines):
					self.mt_color.append([[0, 0, 0]] * l)
					self.mt_height.append([0] * l)
			else:
				# add before
				for _ in range(number_lines):
					self.mt_color.insert(0, [[0, 0, 0]] * l)
					self.mt_height.insert(0, [0] * l)

		for line in matrix:
			x, y, z, *color = map(int, line)
			# y = -y # fliping over the mtx to do it look like a real picture
			x = -x

			expand_x = do_expend(x, 0)
			if (expand_x[0]):
				# we have point with coord tl_corner[0] and we need to have point with coord x
				add_colomns(self.tl_corner[0] - x, 0)
				self.tl_corner[0] = x
			elif (expand_x[1]):
				add_colomns(x - self.lr_corner[0], 1)
				self.lr_corner[0] = x

			expand_y = do_expend(y, 1)
			if (expand_y[0]):
				add_lines(self.tl_corner[1] - y, 0)
				self.tl_corner[1] = y
			elif (expand_y[1]):
				add_lines(y - self.lr_corner[1], 1)
				self.lr_corner[1] = y

			self.mt_color[y - self.tl_corner[1]][x - self.tl_corner[0]] = color
			self.mt_height[y - self.tl_corner[1]][x - self.tl_corner[0]] = abs(z)
		self.HEIGHT, self.WIDTH = len(self.mt_color), len(self.mt_color[0])

	def improve_mt(self, matrix):
		for line in matrix:
			x, y, z, *color = map(int, line)
			# y = -y # fliping over the mtx to do it look like a real picture
			x = -x
		
			mt_y = y - self.tl_corner[1]
			mt_x = x - self.tl_corner[0]
			if ((0 <= mt_y < self.HEIGHT) and (0 <= mt_x < self.WIDTH) and self.mt_height[mt_y][mt_x] == 0):
				self.mt_color[mt_y][mt_x] = color
				self.mt_height[mt_y][mt_x] = abs(z)

	def read_img(self):
		self.mt_color = cv2.cvtColor(cv2.imread(self.PATH_TO_IMG), cv2.COLOR_BGR2RGB)
		self.HEIGHT, self.WIDTH = len(self.mt_color), len(self.mt_color[0])

	def trinarise_mtx(self, standart=True):
		"""mtx to red, blue and black colors only.
		"""
		def color(p, standart):
			if (not standart and p == [0, 0, 0]):
				return [255, 255, 255]
			return [255 * self.is_red(p), 0, 255 * self.is_blue(p)]
		self.mt_color = [[color(p, standart) for p in line] for line in self.mt_color]

	def get_area(self, pixel, img=None):
		"""
		- pixel = (y, x);
		"""
		if (img is None):
			img = self.mt_color
		col = img[pixel[0]][pixel[1]]

		visited = set([pixel])
		queue = set([pixel])
		while (queue):
			next_pnts = self.get_next_pnts(queue.pop(), short_set=False)
			for next_pnt in next_pnts:
				if (next_pnt not in visited and img[next_pnt[0]][next_pnt[1]] == col):
					visited.add(next_pnt)
					queue.add(next_pnt)
		return visited

	def clean_img(self, min_area=370):
		"""Removes small areas from the image.
		"""
		visited_total = set() # {(y, x), (y, x), ..}

		for y in range(self.HEIGHT):
			for x in range(self.WIDTH):

				if (self.mt_color[y][x] != [0, 0, 0] and ((y, x) not in visited_total)):
					cur_block_area = self.get_area((y, x))
					visited_total = visited_total | cur_block_area

					if (len(cur_block_area) <= min_area):
						for (y, x) in cur_block_area:
							self.mt_color[y][x] = [0, 0, 0]

	def get_objects(self):
		main_set = set() # {(y, x), (y, x), ..}
		areas = [] # [Object, Object, ...]

		for y in range(self.HEIGHT):
			for x in range(self.WIDTH):
				if (self.mt_color[y][x] != [0, 0, 0] and ((y, x) not in main_set)):
					visited, cur_block_area = self.get_area((y, x), self.mt_color[y][x])
					main_set = main_set | visited
					if len(cur_block_area) < 400:
						for (y, x) in visited:
							self.mt_color[y][x] = [0, 0, 0]
					else:
						h = [self.mt_height[p[0]][p[1]] for p in cur_block_area]
						img = [[[0, 0, 0] for _ in line] for line in self.mt_color]
						for p in cur_block_area:
							img[p[0]][p[1]] = [255, 0, 0]
						areas.append(Object(cur_block_area, h, img))
		self.objects = areas
		return areas

	def get_corners(self, sorted_contour):
		reds_list = []
		for y, x in sorted_contour:
			area = self.create_area(y, x)
			reds = 0
			for y1, x1 in area:
				if self.mt_color[y1][x1][0] == 255:
					reds += 1
			reds_list.append([reds, (y, x), set(area)])
		reds_list.sort(key=lambda x: x[0])

		remembered_points = set()
		corners = []
		i = 0
		while len(corners) < 4:
			y, x = reds_list[i][1]
			while (y, x) in remembered_points:
				i += 1
				y, x = reds_list[i][1]

			corners.append([y, x])
			remembered_points = remembered_points | reds_list[i][2]
			i += 1
		return corners

	def visualize(self, ignore_colors=[]):
		import open3d as o3d
		x, y, z = [], [], []
		cr, cg, cb = [], [], []
		for i in range(self.HEIGHT):
			for j in range(self.WIDTH):

				if(self.mt_height[i][j] == 0): continue
				if(self.mt_color[i][j] in ignore_colors): continue

				x.append(j)
				y.append(i)
				z.append(self.mt_height[i][j])

				cr.append(self.mt_color[i][j][0])
				cg.append(self.mt_color[i][j][1])
				cb.append(self.mt_color[i][j][2])

		points = np.vstack((x, y, z)).transpose()
		colors = np.vstack((cr, cg, cb)).transpose()

		pcd = o3d.geometry.PointCloud()
		pcd.points = o3d.utility.Vector3dVector(points)
		pcd.colors = o3d.utility.Vector3dVector(colors/255)
		o3d.visualization.draw_geometries([pcd])

	def visualize_points(self, points, ignore_colors=[]):
		import open3d as o3d
		x, y, z = [], [], []
		cr, cg, cb = [], [], []
		for i ,j in points:
			if(self.mt_height[i][j] == 0): continue
			if(self.mt_color[i][j] in ignore_colors): continue

			x.append(j)
			y.append(i)
			z.append(self.mt_height[i][j])

			cr.append(self.mt_color[i][j][0])
			cg.append(self.mt_color[i][j][1])
			cb.append(self.mt_color[i][j][2])

		points = np.vstack((x, y, z)).transpose()
		colors = np.vstack((cr, cg, cb)).transpose()

		pcd = o3d.geometry.PointCloud()
		pcd.points = o3d.utility.Vector3dVector(points)
		pcd.colors = o3d.utility.Vector3dVector(colors/255)
		o3d.visualization.draw_geometries([pcd])

	def determine_blocks(self, method="heights"):
		if (method == "heights"):
			hgts = [[[0, 650 - self.mt_height[y][x], 0] if self.mt_color[y][x] != [0, 0, 0] else [0, 0, 0] for x in range(self.WIDTH)] for y in range(self.HEIGHT)]
			src = np.array(hgts, dtype=np.uint8)
		else:
			src = np.array(self.mt_color, dtype=np.uint8)
		# src = np.array(self.mt_color, dtype=np.uint8)

		gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

		bw = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]

		dist = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
		cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)

		_, dist = cv2.threshold(dist, 0.4, 1.0, cv2.THRESH_BINARY)

		kernel = np.ones((3, 3), dtype=np.uint8)
		dist = cv2.dilate(dist, kernel)

		dist_8u = dist.astype('uint8')
		# Find total markers
		contours, _ = cv2.findContours(dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		# Create the marker image for the watershed algorithm
		markers = np.zeros(dist.shape, dtype=np.int32)
		for i in range(len(contours)):
			cv2.drawContours(markers, contours, i, (i+1), -1)

		cv2.circle(markers, (5, 5), 3, (255, 255, 255), -1)

		cv2.watershed(src, markers)

		# result image
		blocks = {}
		for i in range(markers.shape[0]):
			for j in range(markers.shape[1]):
				cur_index = markers[i, j]
				if (0 < cur_index <= len(contours) and self.mt_color[i][j] != [0, 0, 0]):
					if (cur_index not in blocks):
						blocks[cur_index] = [(i, j)]
					else:
						blocks[cur_index].append((i, j))
		return blocks

	def create_objects(self, method="heights", min_area=370):
		obj = []
		blocks = self.determine_blocks(method)
		for block in blocks.values():
			if len(block) < min_area: continue
			h = {}
			for y, x in block:
				h[(y, x)] = self.mt_height[y][x]
			color = self.most_common_color(block)
			img = np.zeros((self.HEIGHT, self.WIDTH, 3))
			for p in block:
				img[p[0]][p[1]] = color
			obj.append(Object(block, h, img, color))
		return obj

	def find_max_min_blocks(self, blocks):
		# Returns minobj, maxobj sorted by `y` coordinate
		miny, maxy = int(10e10), int(-10e10)
		minobj, maxobj = None, None
		for block in blocks:
			for p in block.area:
				if p[0] < miny:
					miny = p[0]
					minobj = block
				if p[0] > maxy:
					maxy = p[0]
					maxobj = block
		return minobj, maxobj

	def get_rot_angle(self, obj):
		img = np.array(obj.source_img, dtype=np.uint8)
		img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		contours = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

		cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]
		rect = cv2.minAreaRect(cnt)
		box = cv2.boxPoints(rect)
		box = np.int0(box)

		center = np.round(sum(box) / 4)
		cx, cy = map(int, center)

		yx_box = [[y, x] for x, y in list(box)]
		sorted_corners = self.sort_clockwise(yx_box, [cy, cx])

		y_low_coord = max(sorted_corners, key=lambda x: x[0])

		ind = sorted_corners.index(y_low_coord) # find lowest point
		sorted_corners = sorted_corners[ind:] + sorted_corners[:ind]

		a = self.dist(sorted_corners[0], sorted_corners[1])
		b = self.dist(sorted_corners[0], sorted_corners[3])
		if (a > b):
			# tilted to the left
			angle = degrees(acos((sorted_corners[1][1] - y_low_coord[1]) / a))
		else:
			# tilted to the right
			angle = degrees(acos((sorted_corners[3][1] - y_low_coord[1]) / b))
		return angle


def get_quater(y, x):
    if (y < 0 and x < 0):
        return 4
    elif (y > 0 and x < 0):
        return 1
    elif (y < 0 and x > 0):
        return 3
    elif (y > 0 and x > 0):
        return 2

mt = MatrixProcessing()

frame = mt.get_pic_from_file(r"input.ply")
mtx = np.concatenate((np.asarray(frame.points) * 1000, np.asarray(frame.colors) * 255), axis=1)
mt.build_mt(mtx)
mt.trinarise_mtx()
mt.clean_img()


objects = mt.create_objects()


def solve(objects):
	if (len(objects) == 1):
		return get_quater(objects[0].centery + mt.tl_corner[1], objects[0].centerx + mt.tl_corner[0])

	ans = []
	for obj in objects:
		if (cv2.contourArea(obj.box) < 1900 or cv2.contourArea(obj.box) > 3200 or obj.color == [255, 0, 0]):
			pass
		else:
			sorted_corners = mt.sort_clockwise(obj.box)
			a = mt.dist(sorted_corners[0], sorted_corners[1])
			b = mt.dist(sorted_corners[1], sorted_corners[2])
			c = mt.dist(sorted_corners[2], sorted_corners[3])
			d = mt.dist(sorted_corners[3], sorted_corners[0])
			w = (a + c) / 2
			h = (b + d) / 2
			
			if (min(w, h) / max(w, h) < 0.55):
				continue

			ans.append(get_quater(obj.centery + mt.tl_corner[1], obj.centerx + mt.tl_corner[0]))
	if (len(ans) == 0):
		return 4
	if (len(ans) == 1):
		return ans[0]
	else: 
		return ans[1] # len(ans) = 2

print(solve(objects))
