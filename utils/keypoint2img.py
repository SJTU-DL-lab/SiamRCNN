import os.path
from PIL import Image
import numpy as np
import json
import glob
from scipy.optimize import curve_fit
import warnings
# codes for debug

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import skimage.io as io
import pylab
pylab.rcParams['figure.figsize'] = (15, 12.0)
from pycocotools.coco import COCO
%matplotlib inline

def func(x, a, b, c):
    return a * x**2 + b * x + c

def linear(x, a, b):
    return a * x + b

def setColor(im, yy, xx, color):
    if len(im.shape) == 3:
        if (im[yy, xx] == 0).all():
            im[yy, xx, 0], im[yy, xx, 1], im[yy, xx, 2] = color[0], color[1], color[2]
        else:
            im[yy, xx, 0] = ((im[yy, xx, 0].astype(float) + color[0]) / 2).astype(np.uint8)
            im[yy, xx, 1] = ((im[yy, xx, 1].astype(float) + color[1]) / 2).astype(np.uint8)
            im[yy, xx, 2] = ((im[yy, xx, 2].astype(float) + color[2]) / 2).astype(np.uint8)
    else:
        im[yy, xx] = color[0]

def drawEdge(im, x, y, bw=1, color=(255,255,255), draw_end_points=False):
    if x is not None and x.size:
        h, w = im.shape[0], im.shape[1]
        # edge
        for i in range(-bw, bw):
            for j in range(-bw, bw):
                yy = np.maximum(0, np.minimum(h-1, y+i))
                xx = np.maximum(0, np.minimum(w-1, x+j))
                setColor(im, yy, xx, color)

        # edge endpoints
        if draw_end_points:
            for i in range(-bw*2, bw*2):
                for j in range(-bw*2, bw*2):
                    if (i**2) + (j**2) < (4 * bw**2):
                        yy = np.maximum(0, np.minimum(h-1, np.array([y[0], y[-1]])+i))
                        xx = np.maximum(0, np.minimum(w-1, np.array([x[0], x[-1]])+j))
                        setColor(im, yy, xx, color)

def interpPoints(x, y):
    if abs(x[:-1] - x[1:]).max() < abs(y[:-1] - y[1:]).max():
        curve_y, curve_x = interpPoints(y, x)
        if curve_y is None:
            return None, None
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if len(x) < 3:
                popt, _ = curve_fit(linear, x, y)
            else:
                popt, _ = curve_fit(func, x, y)
                if abs(popt[0]) > 1:
                    return None, None
        if x[0] > x[-1]:
            x = list(reversed(x))
            y = list(reversed(y))
        curve_x = np.linspace(x[0], x[-1], (x[-1]-x[0]))
        if len(x) < 3:
            curve_y = linear(curve_x, *popt)
        else:
            curve_y = func(curve_x, *popt)
    return curve_x.astype(int), curve_y.astype(int)

def read_keypoints(json_input, size, random_drop_prob=0, remove_face_labels=False, basic_point_only=False):
    with open(json_input, encoding='utf-8') as f:
        keypoint_dicts = json.loads(f.read())["people"]

    edge_lists = define_edge_lists(basic_point_only)
    w, h = size
    pose_img = np.zeros((h, w, 3), np.uint8)
    for keypoint_dict in keypoint_dicts:
        pose_pts = np.array(keypoint_dict["pose_keypoints_2d"]).reshape(25, 3)
        face_pts = np.array(keypoint_dict["face_keypoints_2d"]).reshape(70, 3)
        hand_pts_l = np.array(keypoint_dict["hand_left_keypoints_2d"]).reshape(21, 3)
        hand_pts_r = np.array(keypoint_dict["hand_right_keypoints_2d"]).reshape(21, 3)
        pts = [extract_valid_keypoints(pts, edge_lists) for pts in [pose_pts, face_pts, hand_pts_l, hand_pts_r]]
        pose_img += connect_keypoints(pts, edge_lists, size, random_drop_prob, remove_face_labels, basic_point_only)
    return pose_img

def extract_valid_keypoints(pts, edge_lists, thre=0.01):
    pose_edge_list, _ = edge_lists
    p = pts.shape[0]
    output = np.zeros((p, 2))

    valid = (pts[:, 2] > thre)
    output[valid, :] = pts[valid, :2]

    return output

def connect_keypoints(pts, edge_lists, size, output_edges):
    pose_pts = pts
    w, h = size
    # output_edges = np.zeros((h, w, 3), np.uint8)
    pose_edge_list, pose_color_list = edge_lists

    ### pose
    for i, edge in enumerate(pose_edge_list):
        x, y = pose_pts[edge, 0], pose_pts[edge, 1]
        if (0 not in x):
            curve_x, curve_y = interpPoints(x, y)
            drawEdge(output_edges, curve_x, curve_y, bw=3, color=pose_color_list[i], draw_end_points=True)

    return output_edges

def define_edge_lists():
    ### pose
    pose_edge_list = []
    pose_color_list = []

    pose_edge_list += [
        [ 0,  1], [14, 16], [ 0, 14], [ 0, 15], [15, 17],           # head
        [ 1,  8], [ 1, 11],                                         # body
        [ 1,  2], [ 2,  3], [ 3,  4],                               # right arm
        [ 1,  5], [ 5,  6], [ 6,  7],                               # left arm
        [ 8,  9], [ 9, 10],                                         # right leg
        [ 11, 12], [12, 13]                                         # left leg
    ]
    pose_color_list += [
        [153,  0, 51], [153,  0,153], [153,  0,102], [102,  0,153], [ 51,  0,153],
        [  0,153, 51], [  0,102,153],
        [153, 51,  0], [153,102,  0], [153,153,  0],
        [102,153,  0], [ 51,153,  0], [  0,153,  0],
        [  0,153,102], [  0,153,153],
        [  0, 51,153], [  0,  0,153]
    ]

    return pose_edge_list, pose_color_list

def mean_pos(pos1, pos2):
    m_pos = pos1 + (pos2 - pos1) / 2
    return m_pos
# coco: 0-nose    1-Leye    2-Reye    3-Lear    4-Rear
# 5-Lsho    6-Rsho    7-Lelb    8-Relb    9-Lwri    10-Rwri
# 11-Lhip    12-Rhip    13-Lkne    14-Rkne    15-Lank    16-Rank

# openpose: 0-'nose', 1-'neck', 2-'Rsho', 3-'Relb', 4-'Rwri'
# 5-'Lsho', 6-'Lelb', 7-'Lwri', 8-'Rhip', 9-'Rkne', 10-'Rank'
# 11-'Lhip', 12-'Lkne', 13-'Lank', 14-'Leye', 15-'Reye',
# 16-'Lear', 17-'Rear', 18-'pt19'
def coco_to_openpose(coco_kp):
    # coco_kp shape: [17, 3]
    # output shape: [18, 3]
    opose_kp = np.zeros((18, 3), dtype=coco_kp.dtype)
    neck = mean_pos(coco_kp[5], coco_kp[6])
    opose_kp[0] = coco_kp[0]
    opose_kp[1] = neck
    opose_kp[2] = coco_kp[6]
    opose_kp[3] = coco_kp[8]
    opose_kp[4] = coco_kp[10]
    opose_kp[5] = coco_kp[5]
    opose_kp[6] = coco_kp[7]
    opose_kp[7] = coco_kp[9]
    opose_kp[8] = coco_kp[12]
    opose_kp[9] = coco_kp[14]
    opose_kp[10] = coco_kp[16]
    opose_kp[11] = coco_kp[11]
    opose_kp[12] = coco_kp[13]
    opose_kp[13] = coco_kp[15]
    opose_kp[14:] = coco_kp[1:5]

    return opose_kp

edge_lists = define_edge_lists()
def coco_pose_to_img(coco_pose, img, size, thresh=0.2):
    # coco_pose shape: [17, 3]
    # size: [h, w]
    # bs = coco_pose.shape[0]
    h, w = size

    opose_pts = coco_to_openpose(coco_pose)
    pts = extract_valid_keypoints(opose_pts, edge_lists, thresh)
    img = connect_keypoints(pts, edge_lists, [w, h], img)
    return img

if __name__ == '__main__':

    annFile = '/home/yaosy/Diskb/projects/OSIS/CenterNet-AUT/OSISnet/data/coco/annotations/person_keypoints_train2017.json'
    coco=COCO(annFile)
    # display COCO categories and supercategories
    cats = coco.loadCats(coco.getCatIds())
    nms=[cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(nms)))

    nms = set([cat['supercategory'] for cat in cats])
    print('COCO supercategories: \n{}'.format(' '.join(nms)))
    catIds = coco.getCatIds(catNms=['person'])
    imgIds = coco.getImgIds(catIds=catIds)
    img = coco.loadImgs(imgIds[0])[0]
    I = io.imread(img['coco_url'])
    plt.axis('off')
    plt.imshow(I)
    ax = plt.gca()
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns)
    kp = anns[0]['keypoints']

    pose_pts = np.array(kp).reshape(17, 3)
    opose_pts = coco_to_openpose(pose_pts)
    edge_lists = define_edge_lists()
    pts = extract_valid_keypoints(opose_pts, edge_lists)
    h, w = img['height'], img['width']
    output = connect_keypoints(pts, edge_lists, [w, h], I)

    # plt.axis('off')
    plt.imshow(output)
    plt.show()
