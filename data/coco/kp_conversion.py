#author:Kevin  time:2019/6/17
import numpy as np

skeleton = [[1, 1], [1, 2], [1, 3], [1, 4], [1, 5],
            [2, 1], [2, 2], [2, 4], [2, 4], [2, 5],
            [3, 1], [3, 2], [3, 3], [3, 4], [3, 5],
            [4, 1], [4, 2]]
matrix = np.array([[1, 2, 3], [4, 5, 6]])


'''skeleton是从json文件导出的骨架点list， matrix是转化矩阵，用源程序中的就可以'''


def kp_conversion(skeleton, matrix):

    key_points = []
    kps_conversion = []

    for i in range(0, len(skeleton)):
        lis = skeleton[i]
        lis.append(1)
        key_points.append(lis)
        # print(key_points)

    key_points = np.array(key_points)

    for i in range(0, int(key_points.shape[0])):
        ky_conversion = np.matmul(matrix, key_points[i, :]).tolist()
        kps_conversion.append(ky_conversion)

    return kps_conversion


kps_conversion = kp_conversion(skeleton, matrix)
print(kps_conversion)