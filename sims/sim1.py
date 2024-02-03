"""Определяет цвета блоков с самой большой и самой маленькой
координатой по оси 𝑌 любой части блока, получая на вход облако точек.

Входные данные: через файл input.ply управляющей программе передается облако точек.
Ожидаемый результат: В результате работы программы в консоль выведено 2 строки:
- Цвет блока с наибольшей координатой центра
- Цвет блока с наименьшей координатой центра
В качестве цвета необходимо вывести “RED” или “BLUE”.
"""


from math import degrees, sqrt, acos
import numpy as np
import cv2
import open3d as o3d


class Tools:
	def __init__(self) -> None:
		self.red_low_1 = [0, 0.4, 0.35]
		self.red_high_1 = [23, 1, 1]
		self.red_low_2 = [340,0.4, 0.4]
		self.red_high_2 = [360, 1, 1]

		self.blue_low = [185, 0.5, 0.2]
		self.blue_high = [245, 1, 1]

	def hex2rgb(self, pixel):
		return [int(pixel[0:2], 16), int(pixel[2:4], 16), int(pixel[4:6], 16)]
	def rgb2hsv(self, pixel):
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
	def get_color_rgb(self, pixel):
		if (self.is_red(pixel)):
			return [255, 0, 0]
		elif (self.is_blue(pixel)):
			return [0, 0, 255]
		else:
			# print("Color is not red and blue")
			return [0, 0, 0]
	def most_common_color(self, area):
		red = 0
		blue = 0
		for (y, x) in area:
			color = self.get_color_rgb(self.mtx_cols[y][x])
			if (color == [255, 0, 0]):
				red += 1
			elif (color == [0, 0, 255]):
				blue += 1
		if (red > blue):
			return [255, 0, 0]
		else:
			return [0, 0, 255]

	def dist(self, p1, p2):
		y1, x1 = p1
		y2, x2 = p2
		return sqrt((y2 - y1) * (y2 - y1) + (x2 - x1) * (x2 - x1))

	def get_next_pnts(self, curPnt, short_set=True, allow_height=True):
		nextPnts = []

		for sx in range(-1, 2):
			for sy in range(-1, 2):
				if (short_set and sx * sx + sy * sy != 1): continue #Skip if short test enable
				if (sx * sx + sy * sy == 0): continue  # Skip if (curpnt[0] + 0, curpnt[0] + 0)
				to = (curPnt[0] + sy, curPnt[1] + sx)
				if not (0 <= to[0] < self.HEIGHT and 0 <= to[1] < self.WIDTH): continue #Skip if pixel out of range
				if (allow_height and abs(self.mtx_hgts[curPnt[0]][curPnt[1]] - self.mtx_hgts[to[0]][to[1]]) < 2):
					nextPnts.append(to)
				elif (not allow_height):
					nextPnts.append(to)
		return nextPnts

	def create_area(self, y, x):
		if (x >= 10 and x + 10 < self.WIDTH and y >= 10 and y + 10 < self.HEIGHT):
			return [(y + yc, x + xc) for xc in range(-10, 11) for yc in range(-10, 11)]
		area = []
		for xc in range(0, 10):
			for yc in range(0, 10):
				if (y >= yc and x >= xc):
					area.append((y - yc, x - xc))
			for yc in range(1, 11):
				if (y + yc < self.HEIGHT and x >= xc):
					area.append((y + yc, x - xc))
		for xc in range(1, 11):
			for yc in range(0, 10):
				if (y >= yc and x + xc < self.WIDTH):
					area.append((y - yc, x + xc))
			for yc in range(1, 11):
				if (y + yc < self.HEIGHT and x + xc < self.WIDTH):
					area.append((y + yc, x + xc))
		# return boarder, area
		return area

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

class Object(Tools):
	def __init__(self, points, heights, img, color=[0, 0, 0]) -> None:
		self.area = points #Point [(y, x)]
		self.heights = {}
		for i, point in enumerate(self.area):
			self.heights[point] = heights[i]
		self.img = img
		self.source_img = img
		self.color = color

		self.contourxy = self.get_contourxy()
		self.contourz  = self.get_contourz()

		#self.centery, self.centerx, self.centerz = self.get_center()

		self.normal = self.get_normal()

		self.img = np.array(self.img, dtype=np.uint8)
		self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
		cvcontours = cv2.findContours(self.img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
		cnt = sorted(cvcontours, key=cv2.contourArea, reverse=True)[0]
		rect = cv2.minAreaRect(cnt)
		self.box = cv2.boxPoints(rect)
		self.box = np.int0(self.box)

		center = np.round(sum(self.box) / 4)
		self.centerx, self.centery = map(int, center)
		self.centerz = sum(heights) / len(heights)

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
		test_img = np.array([[list(i) for i in line] for line in self.source_img])
		cv2.drawContours(test_img, [self.box], 0, (0, 255, 0), 1)

class MatrixProcessing(Tools):
	def __init__(self, path=None, path_to_img=None, name=1):
		super().__init__()

		if (path):
			self.PATH = path
		if (path_to_img):
			self.PATH_TO_IMG = path_to_img

		self.mtx_cols = [[[255, 255, 255]]] # begin with this for convenience
		self.mtx_hgts = [[0]]

		self.HEIGHT, self.WIDTH = None, None
		# top left and low right corners
		# we don't care about z
		self.tl_corner = [0, 0] # xl, yu | indexes!!
		self.lr_corner = [0, 0] # xr, yl

		self.objects = []
		self.contour = [] # [(y, x), (y, x), ..]
		self.blocks_data = []

	def built_mtx_ply(self, matrix):
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
				for i in range(len(self.mtx_cols)):
					self.mtx_cols[i].extend([[0, 0, 0]] * number_colomns)
					self.mtx_hgts[i].extend([0] * number_colomns)
			else:
				# add before
				for _ in range(number_colomns):
					for i in range(len(self.mtx_cols)):
						self.mtx_cols[i].insert(0, [0, 0, 0])
						self.mtx_hgts[i].insert(0, 0)
		def add_lines(number_lines, where):
			l = len(self.mtx_cols[0])
			if (where):
				# add after
				for _ in range(number_lines):
					self.mtx_cols.append([[0, 0, 0]] * l)
					self.mtx_hgts.append([0] * l)
			else:
				# add before
				for _ in range(number_lines):
					self.mtx_cols.insert(0, [[0, 0, 0]] * l)
					self.mtx_hgts.insert(0, [0] * l)

		for line in matrix:
			x, y, z = map(int, line[0:3])
			# y = -y # fliping over the mtx to do it look like a real picture

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

			color = list(map(int, line[3:]))
			self.mtx_cols[y - self.tl_corner[1]][x - self.tl_corner[0]] = [255 * self.is_red(color), 0, 255 * self.is_blue(color)]

			self.mtx_hgts[y - self.tl_corner[1]][x - self.tl_corner[0]] = abs(z)
		self.HEIGHT, self.WIDTH = len(self.mtx_cols), len(self.mtx_cols[0])

	def trinarise_mtx(self):
		"""mtx to red, blue and black colors only.
		"""
		self.mtx_cols = [[[255 * self.is_red(p), 0, 255 * self.is_blue(p)] for p in line] for line in self.mtx_cols]

	def spread(self, pixel, col=None, img=None):
		"""
		- pixel = (y, x);
		- col = [r, g, b], (optional, by default, is determined relying on "pixel");
		"""
		if (col is None):
			col = self.get_color_rgb(pixel)
		if (img is None):
			img = self.mtx_cols

		visited = set([pixel])
		queue = set([pixel])
		cur_block_area = [pixel]
		while (queue):
			next_pnts = self.get_next_pnts(queue.pop(), short_set=True)
			for next_pnt in next_pnts:
				if (next_pnt not in visited and img[next_pnt[0]][next_pnt[1]] == col):
					visited.add(next_pnt)
					queue.add(next_pnt)
					cur_block_area.append(next_pnt)
		return visited, cur_block_area

	def clean_img(self):
		"""Removes small areas from the image.
		"""
		visited_total = set() # {(y, x), (y, x), ..}
		# areas = [] # [Object, Object, ...]

		for y in range(self.HEIGHT):
			for x in range(self.WIDTH):

				if (self.mtx_cols[y][x] != [0, 0, 0] and ((y, x) not in visited_total)):
					visited, cur_block_area = self.spread((y, x), self.mtx_cols[y][x])
					visited_total = visited_total | visited

					if (len(cur_block_area) < 370):
						for (y, x) in visited:
							self.mtx_cols[y][x] = [0, 0, 0]

	def get_objects(self):
		main_set = set() # {(y, x), (y, x), ..}
		areas = [] # [Object, Object, ...]

		for y in range(self.HEIGHT):
			for x in range(self.WIDTH):
				if (self.mtx_cols[y][x] != [0, 0, 0] and ((y, x) not in main_set)):
					visited, cur_block_area = self.spread((y, x), self.mtx_cols[y][x])
					main_set = main_set | visited
					if len(cur_block_area) < 400:
						for (y, x) in visited:
							self.mtx_cols[y][x] = [0, 0, 0]
					else:
						h = [self.mtx_hgts[p[0]][p[1]] for p in cur_block_area]
						img = [[[0, 0, 0] for _ in line] for line in self.mtx_cols]
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
				if self.mtx_cols[y1][x1][0] == 255:
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

				if(self.mtx_hgts[i][j] == 0): continue
				if(self.mtx_cols[i][j] in ignore_colors): continue

				x.append(j)
				y.append(i)
				z.append(self.mtx_hgts[i][j])

				cr.append(self.mtx_cols[i][j][0])
				cg.append(self.mtx_cols[i][j][1])
				cb.append(self.mtx_cols[i][j][2])

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
			if(self.mtx_hgts[i][j] == 0): continue
			if(self.mtx_cols[i][j] in ignore_colors): continue

			x.append(j)
			y.append(i)
			z.append(self.mtx_hgts[i][j])

			cr.append(self.mtx_cols[i][j][0])
			cg.append(self.mtx_cols[i][j][1])
			cb.append(self.mtx_cols[i][j][2])

		points = np.vstack((x, y, z)).transpose()
		colors = np.vstack((cr, cg, cb)).transpose()

		pcd = o3d.geometry.PointCloud()
		pcd.points = o3d.utility.Vector3dVector(points)
		pcd.colors = o3d.utility.Vector3dVector(colors/255)
		o3d.visualization.draw_geometries([pcd])

	def determine_blocks(self, image):
		src = np.array(image, dtype=np.uint8)
		# src = cv2.cvtColor(src, cv2.COLOR_RGB2BGR) # optional

		gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

		bw = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]

		dist = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
		cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)

		_, dist = cv2.threshold(dist, 0.4, 1.0, cv2.THRESH_BINARY)

		kernel = np.ones((3,3), dtype=np.uint8)
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
				if (0 < cur_index <= len(contours) and image[i][j] != [0, 0, 0]):
					if (cur_index not in blocks):
						blocks[cur_index] = [(i, j)]
					else:
						blocks[cur_index].append((i, j))
		return blocks

	def create_objects(self):
		obj = []
		blocks = self.determine_blocks(self.mtx_cols)
		for block in blocks.values():
			if len(block) < 370 : continue
			h = [self.mtx_hgts[p[0]][p[1]] for p in block]
			img = [[[0, 0, 0] for _ in line] for line in self.mtx_cols]
			color = self.most_common_color(block)
			for p in block:
				img[p[0]][p[1]] = color
			obj.append(Object(block, h, img, color))
		return obj

	def find_max_min_blocks(self, blocks):
		#Returns minobj, maxobj sorted by `y` coordinate
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


def get_picture_from_file():
	path = r"C:\Users\Admin\Desktop\Mine\PROGRAMMING\Olympiads\NTO2021-22\tests\open-tests\1.ply"
	# path = r"input.ply"
	pcd = o3d.io.read_point_cloud(path)

	# Visualizing test data frame
	# cam.visualization_of_points(pcd)
	return pcd

def color_name(color):
	if (color == [255, 0, 0]):
		return "RED"
	elif (color == [0, 0, 255]):
		return "BLUE"
	else:
		return "UNKNOWN COLOR"

def main():
	frame = get_picture_from_file()

	arr = np.concatenate((np.asarray(frame.points) * 1000, np.asarray(frame.colors) * 255), axis=1)

	mt = MatrixProcessing()
	mt.built_mtx_ply(arr)

	mt.clean_img()

	objects = mt.create_objects()
	minobj, maxobj = mt.find_max_min_blocks(objects)

	print(color_name(minobj.color))
	print(color_name(maxobj.color))


if __name__ == "__main__":
	main()
