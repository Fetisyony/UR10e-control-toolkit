import bdb
from itertools import count
from dots3d import MatrixProcessing, Object
from OperateCamera import OperateCamera
import numpy as np
import open3d as o3d
import cv2
import random as rng
import time
import sys

import time


class FrameProcessing():
	def __init__(self, cameraon=True, pathnpy=r'', pathply=r''):
		"""
			camera=True			брать изображение с камеры
			camera=False		брать изображение из pathnpy и pathply
			----------------------------------------------------------------------------------------
			arr_cols			необработанная фотка прям с камеры
			arr_dots			список точек в СО камеры x,y,z
			mtx_cols			уменьшенная фотка (красный синий черный) (высота, ширина, rgb)
			mtx_global_hgts		матрица глобальных координат  (высота, ширина, yxz) в мм
			objects				список из Object
			dst					уменьшенная цветная фотка с сегментированными блоками
		"""
		self.mtx_cols=np.zeros((1), dtype=np.uint8)
		self.mtx_global_hgts=np.zeros((1), dtype=np.int32)
		self.objects=[]
		self.dst=np.zeros((1), dtype=np.uint8)
		if cameraon:
			self.cam = OperateCamera(resolution=1)
			pcd, color_image, _ = self.cam.catch_frame()
			self.cam.stop()
			self.arr_cols = cv2.cvtColor(np.array(color_image), cv2.COLOR_BGR2RGB)
			self.arr_dots = (np.array(pcd.points) * 1000).astype(np.int32)	#сырые значения
		else:
			self.arr_cols = cv2.cvtColor(np.load(pathnpy), cv2.COLOR_BGR2RGB)
			self.arr_dots = (np.array(o3d.io.read_point_cloud(pathply).points)*1000).astype(np.int32)

	#=====================матрица из облака точек [y,x,z] глобальные
	def build_mtx_hgts(self, arr_dots):
		mtx_hgts = np.zeros((720, 1280, 3), dtype=np.int32)

		for i in range(len(arr_dots)):
			x = arr_dots[i][0]
			y = arr_dots[i][1]
			z = np.abs(arr_dots[i][2])
			mtx_hgts[360-y][640+x] = [-200-x, -800+y, z]

		sumx = np.sum(mtx_hgts, axis=0)
		sumy = np.sum(mtx_hgts, axis=1)
		up, down, left, right = -1, -1, -1, -1
		for i in range(640):
			if up == -1 and sumy[i][2] > 0:
				up = i
			if down == -1 and sumy[-i][2] > 0:
				down = 720-i
			if left == -1 and sumx[i][2] > 0:
				left = i
			if right == -1 and sumx[-i][2] > 0:
				right = 1280-i
		
		_, mtx_hgts, _ = np.split(mtx_hgts, [up, down], axis=0)
		_, mtx_hgts, _ = np.split(mtx_hgts, [left, right], axis=1)

		#cv2.imshow('p', np.split(mtx_hgts.astype(np.uint8), 3, axis=2)[2])

		return mtx_hgts
	#=====================матрица из облака точек [y,x,z] глобальные


	#=====================красный синий черный
	def trinarize(self, mtx_cols):
		# удаление белого фона
		lower = np.array([0, 110, 0])
		upper = np.array([255, 255, 255])
		hsv = cv2.cvtColor(mtx_cols, cv2.COLOR_RGB2HSV)
		mask = cv2.inRange(hsv, lower, upper)
		output = cv2.bitwise_and(mtx_cols, mtx_cols, mask=mask)
		# cv2.imshow('iiii',output)
		#cv2.imwrite('outtt.png', output)
		# тринаризация изображения
		hsv = cv2.cvtColor(output, cv2.COLOR_RGB2HSV)
		bd = np.array([40, 0, 0])
		bu = np.array([179, 255, 255])
		rd = np.array([0, 0, 0])
		ru = np.array([70, 255, 255])
		for y in range(output.shape[0]):
			for x in range(output.shape[1]):
				# print(output[y][x])
				if (hsv[y][x] > bd).all() and (hsv[y][x] < bu).all():
					output[y][x] = [0, 0, 255]
				elif (hsv[y][x] > rd).all() and (hsv[y][x] < ru).all():
					output[y][x] = [255, 0, 0]
				else:
					output[y][x] = [0, 0, 0]
		mtx_cols = output
		#cv2.imshow('i',mtx_cols)
		
		return mtx_cols
	#=====================красный синий черный

	#=====================выделение блоков по высоте - не используется
	def determine_blocks_by_heght(self, image, hights):
		hgts = [[[0, 650-hights[y][x], 0] if not (image[y][x] == [0, 0, 0]).all(
		) else [0, 0, 0] for x in range(image.shape[1])] for y in range(image.shape[0])]
		hgts = np.array(hgts, dtype=np.uint8)

		blocks = self.determine_blocks(hgts)
		return blocks
	#=====================выделение блоков по высоте - не используется

	#=====================выделение блоков по цвету
	def determine_blocks(self, image):
		#cv2.imshow('original ', image)
		kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
		imgLaplacian = cv2.filter2D(image, cv2.CV_32F, kernel)
		sharp = np.float32(image)
		imgResult = sharp - imgLaplacian
		# convert back to 8bits gray scale
		imgResult = np.clip(imgResult, 0, 255)
		imgResult = imgResult.astype('uint8')
		imgLaplacian = np.clip(imgLaplacian, 0, 255)
		imgLaplacian = np.uint8(imgLaplacian)

		bw = cv2.cvtColor(imgResult, cv2.COLOR_BGR2GRAY)

		dist = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
		# Normalize the distance image for range = {0.0, 1.0}
		# so we can visualize and threshold it
		cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)

		_, dist = cv2.threshold(dist, 0.4, 1.0, cv2.THRESH_BINARY)
		# Dilate a bit the dist image
		kernel1 = np.ones((3, 3), dtype=np.uint8)
		dist = cv2.dilate(dist, kernel1)

		dist_8u = dist.astype('uint8')
		# Find total markers
		contours, _ = cv2.findContours(
			dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		# Create the marker image for the watershed algorithm
		markers = np.zeros(dist.shape, dtype=np.int32)
		# Draw the foreground markers
		for i in range(len(contours)):
			cv2.drawContours(markers, contours, i, (i+1), -1)
		# Draw the background marker
		cv2.circle(markers, (5, 5), 3, (255, 255, 255), -1)

		cv2.watershed(imgResult, markers)

		# result image
		colors = []
		for contour in contours:
			colors.append(
				(rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256)))
		# Create the result image
		dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)
		# Fill labeled objects with random colors
		for i in range(markers.shape[0]):
			for j in range(markers.shape[1]):
				index = markers[i, j]
				if index > 0 and index <= len(contours):
					dst[i, j, :] = colors[index-1]
		self.dst=dst

		blocks = {}
		for i in range(markers.shape[0]):
			for j in range(markers.shape[1]):
				cur_index = markers[i, j]
				if 0 < cur_index <= len(contours):

					# print(cur_index)
					if (cur_index not in blocks):
						blocks[cur_index] = [(i, j)]
					else:
						blocks[cur_index].append((i, j))
		return blocks
	#=====================выделение блоков по цвету

	def most_common_color(self, area, mtx_cols):
		red = 0
		blue = 0
		for (y, x) in area:
			color = mtx_cols[y][x]
			if (color == [0, 0, 255]).all():
				red += 1
			elif (color == [255, 0, 0]).all():
				blue += 1
		if (red > blue):
			return [255, 0, 0]
		else:
			return [0, 0, 255]

	def frameprocessing(self):
		mtx_hgts_ply = self.build_mtx_hgts(self.arr_dots)
		mtx_cols = cv2.resize(
			self.arr_cols, (mtx_hgts_ply.shape[1], mtx_hgts_ply.shape[0]))

		#=====================сжатие матриц в 4 раза
		for i in range(mtx_cols.shape[0]//2):
			mtx_cols = np.delete(mtx_cols, (i), axis=0)
			mtx_hgts_ply = np.delete(mtx_hgts_ply, (i), axis=0)
		for i in range(mtx_cols.shape[1]//2):
			mtx_cols = np.delete(mtx_cols, (i), axis=1)
			mtx_hgts_ply = np.delete(mtx_hgts_ply, (i), axis=1)
		#=====================сжатие матриц в 4 раза

		mtx_cols = self.trinarize(mtx_cols)

		self.mtx_cols = mtx_cols
		self.mtx_global_hgts =mtx_hgts_ply

		blocks = self.determine_blocks(mtx_cols)
		#blocks = determine_blocks_by_heght(mtx_cols, np.split(mtx_hgts_ply,3,axis=2)[2])
		#print(blocks.keys())

		obj = []
		for block in blocks.values():
			if len(block) < 100:
				continue
			h = [mtx_hgts_ply[p[0]][p[1]][2] for p in block]
			y = [mtx_hgts_ply[p[0]][p[1]][0] for p in block]
			x = [mtx_hgts_ply[p[0]][p[1]][1] for p in block]
			img = [[[0, 0, 0] for _ in line] for line in mtx_cols]
			color = self.most_common_color(block, mtx_cols)
			for p in block:
				img[p[0]][p[1]] = color

			obj.append(Object(block, h, img, color))

			obj[-1].globaly = (sum(y) / len([i for i in y if i != 0]))/1000
			obj[-1].globalx = (sum(x) / len([i for i in x if i != 0]))/1000-0.035
			obj[-1].globalz = 0.345	#mtx_hgts_ply[y][x][2]/1000

		self.objects=obj
	
	def calibration(self):
		print('create calibration')
		dots = []
		for i in range(4):
			dots.append([self.objects[i].globaly, self.objects[i].globalx])
			dots.append([self.objects[i].globaly+0.02, self.objects[i].globalx-0.02])
			dots.append([self.objects[i].globaly+0.03, self.objects[i].globalx-0.03])
			dots.append([self.objects[i].globaly+0.04, self.objects[i].globalx-0.04])
			dots.append([self.objects[i].globaly+0.05, self.objects[i].globalx-0.05])
		print('complete')
		return dots