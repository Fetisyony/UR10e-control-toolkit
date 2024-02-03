"""Подсчитывает количество объектов, которые не касаются основания.
Гарантируется что все кубики видны на камере хотя бы частично.

Входные данные: через файл input.ply управляющей программе передается облако точек.
Ожидаемый результат: В результате работы программы в консоль выведено целое число —
количество объектов, которые не касаются основания.
"""


import OperateCamera
import OperateRobot
import open3d as o3d
import cv2
import numpy as np
import time
import random as rng



def build_mtx(arr_cols, arr_dots):
    mtx_hgts = np.zeros((720, 1280, 3), dtype=np.int32)
    mtx_cols = np.zeros((720, 1280, 3), dtype=np.uint8)

    for i in range(len(arr_dots)):

        x = arr_dots[i][0]
        y = arr_dots[i][1]
        z = np.abs(arr_dots[i][2])
        mtx_hgts[360-y][640+x] = [x, y, z]
        mtx_cols[360-y][640+x] = arr_cols[i]

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

    _, mtx_cols, _ = np.split(mtx_cols, [up, down], axis=0)
    _, mtx_cols, _ = np.split(mtx_cols, [left, right], axis=1)
    _, mtx_hgts, _ = np.split(mtx_hgts, [up, down], axis=0)
    _, mtx_hgts, _ = np.split(mtx_hgts, [left, right], axis=1)

    mtx_cols = cv2.cvtColor(mtx_cols, cv2.COLOR_RGB2BGR)

    return mtx_cols, mtx_hgts


def trinarize(mtx_cols):
    """Красный - синий - черный
    """
    # удаление белого фона
    lower = np.array([0, 110, 0])
    upper = np.array([255, 255, 255])
    hsv = cv2.cvtColor(mtx_cols, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(mtx_cols, mtx_cols, mask=mask)

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

    return mtx_cols

def determine_blocks_by_heght(image, hights):
    """Выделение блоков
    """
    _, _, hights = np.split(hights, 3, axis=2)

    minh = min([i for i in hights.flatten() if i != 0])
    maxh = max([i for i in hights.flatten()])
    # print(minh, maxh)
    # hights[y][x]-minh<10 нижний уровень
    # print(image.shape)
    hgts = np.zeros(image.shape, dtype=np.uint8)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if not (image[y][x] == [0, 0, 0]).all() and hights[y][x]-minh < 5:
                hgts[y][x][1] = hights[y][x]

    if maxh-minh < 22:
        return {}
    else:
        blocks = determine_blocks(hgts)
        return blocks

def dilatation(src):
    dilatation_size = 2
    dilation_shape = cv2.MORPH_ELLIPSE  # cv2.MORPH_RECT#
    element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                        (dilatation_size, dilatation_size))
    dilatation_dst = cv2.dilate(src, element)
    return dilatation_dst


def determine_blocks(image):
    """blocks by color
    """
    image = dilatation(image)
    # cv2.imwrite('out.png', image)
    # cv2.imshow('original ', image)
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

    # _, bw = cv.threshold(bw, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # cv2.imshow('Binary Image2', bw)
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 3)

    # Normalize the distance image for range = {0.0, 1.0}
    # so we can visualize and threshold it
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)

    # cv2.imshow('Distance Transform Image', dist)
    _, dist = cv2.threshold(dist, 0.4, 1.0, cv2.THRESH_BINARY)

    # Dilate a bit the dist image
    kernel1 = np.ones((3, 3), dtype=np.uint8)
    dist = cv2.dilate(dist, kernel1)

    # cv2.imshow('Peaks', dist)
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
    markers_8u = (markers * 10).astype('uint8')
    # cv2.imshow('Markers', markers_8u)
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
    # =================================================
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


def most_common_color(area, mtx_cols):
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


def main():
    np.set_printoptions(threshold=np.inf)

    pcd = o3d.io.read_point_cloud(r"input.ply")
    arr_cols = (np.array(pcd.colors) * 255).astype(np.uint8)
    arr_dots = (np.array(pcd.points) * 500).astype(np.int32)

    mtx_cols, mtx_hgts = build_mtx(arr_cols, arr_dots)
    # print(np.split(mtx_hgts,3,axis=2)[2])
    mtx_cols = trinarize(mtx_cols)
    # blocks = determine_blocks(mtx_cols)
    blocks = determine_blocks_by_heght(mtx_cols, mtx_hgts)

    objs = []
    for block in blocks.values():
        if len(block) < 180:
            continue
        h = [mtx_hgts[p[0]][p[1]][2] for p in block]
        y = [mtx_hgts[p[0]][p[1]][0] for p in block]
        x = [mtx_hgts[p[0]][p[1]][1] for p in block]
        img = [[[0, 0, 0] for _ in line] for line in mtx_cols]
        color = most_common_color(block, mtx_cols)
        for p in block:
            img[p[0]][p[1]] = color

        objs.append(Object(block, h, img, color))

    print(len(objs))


if __name__ == "__main__":
    main()
