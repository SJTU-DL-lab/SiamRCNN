# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
from pycocotools.coco import COCO
from os.path import join
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2

dataDir = '.'


def crop_hwc(bbox, out_sz=511):
    a = (out_sz - 1) / (bbox[2] - bbox[0])
    b = (out_sz - 1) / (bbox[3] - bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    # crop = cv2.warpAffine(image, mapping, (out_sz, out_sz),
    # borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return mapping

def crop_hwc1(image, bbox, out_sz, padding=(0, 0, 0)):
    a = (out_sz-1) / (bbox[2]-bbox[0])
    b = (out_sz-1) / (bbox[3]-bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz),
                          borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return crop


def pos_s_2_bbox(pos, s):
    bbox = [pos[0] - s / 2, pos[1] - s / 2, pos[0] + s / 2, pos[1] + s / 2]
    return bbox


def crop_like_SiamFCx(bbox, exemplar_size=127, context_amount=0.5, search_size=255):
    target_pos = [(bbox[2] + bbox[0]) / 2., (bbox[3] + bbox[1]) / 2.]
    target_size = [bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1]
    wc_z = target_size[1] + context_amount * sum(target_size)
    hc_z = target_size[0] + context_amount * sum(target_size)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = exemplar_size / s_z
    d_search = (search_size - exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    # x = crop_hwc1(image, pos_s_2_bbox(target_pos, s_x), search_size, padding)
    return target_pos, s_x


def crop_like_SiamFCx1(image, bbox, exemplar_size=127, context_amount=0.5, search_size=255, padding=(0, 0, 0)):
    target_pos = [(bbox[2]+bbox[0])/2., (bbox[3]+bbox[1])/2.]
    target_size = [bbox[2]-bbox[0]+1, bbox[3]-bbox[1]+1]
    wc_z = target_size[1] + context_amount * sum(target_size)
    hc_z = target_size[0] + context_amount * sum(target_size)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = exemplar_size / s_z
    d_search = (search_size - exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    x = crop_hwc1(image, pos_s_2_bbox(target_pos, s_x), search_size, padding)
    return x


def kp_conversion(KeyPoints, matrix):

    key_points = []
    kps_conversion = []
    skeleton = [0, 0]
    Skeleton = []

    for i in range(0, int(len(KeyPoints)/3)):
        skeleton[0] = KeyPoints[i * 3 + 0]
        skeleton[1] = KeyPoints[i * 3 + 1]
        Skeleton.append(skeleton[:])
        lis = Skeleton[i]
        lis.append(1)
        key_points.append(lis)

    key_points = np.array(key_points)

    for i in range(0, int(len(KeyPoints)/3)):
        if KeyPoints[i * 3 + 2] != 0:
            ky_conversion = np.matmul(matrix, key_points[i, :]).tolist()
            kps_conversion.append(ky_conversion[0])
            kps_conversion.append(ky_conversion[1])
            kps_conversion.append(KeyPoints[i * 3 + 2])
        else:
            kps_conversion.append(0)
            kps_conversion.append(0)
            kps_conversion.append(0)

    return kps_conversion


for data_subset in ['val2017', 'train2017']:
# for data_subset in ['val2017']:
    dataset = dict()
    # annFile = '{}/annotations/instances_{}.json'.format(dataDir, data_subset)
    annFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir, data_subset)
    coco = COCO(annFile)
    n_imgs = len(coco.imgs)

    # deal with class names
    cats = [cat['name']
            for cat in coco.loadCats(coco.getCatIds())]
    classes = ['__background__'] + cats
    num_classes = len(classes)
    class_to_ind = dict(zip(classes, range(num_classes)))
    class_to_coco_ind = dict(zip(cats, coco.getCatIds()))
    coco_ind_to_class_ind = dict([(class_to_coco_ind[cls],
                                   class_to_ind[cls])
                                  for cls in classes[1:]])

    for n, img_id in enumerate(coco.imgs):
        print('subset: {} image id: {:04d} / {:04d}'.format(data_subset, n, n_imgs))
        img = coco.loadImgs(img_id)[0]
        annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        anns = coco.loadAnns(annIds)
        crop_base_path = join(data_subset, img['file_name'].split('/')[-1].split('.')[0])
        set_img_base_path = join(dataDir, data_subset)
        im = cv2.imread('{}/{}'.format(set_img_base_path, img['file_name']))
        # print('{}/{}'.format(set_img_base_path, img['file_name']))

        if len(anns) > 0:
            dataset[crop_base_path] = dict()

        for track_id, ann in enumerate(anns):

            cls = coco_ind_to_class_ind[ann['category_id']]
            if cls != 1:
                continue

            # ignore objs without keypoints annotation
            if max(ann['keypoints']) == 0:
                continue

            rect = ann['bbox']
            if rect[2] <= 0 or rect[3] <= 0:  # lead nan error in cls.
                continue
            bbox = [rect[0], rect[1], rect[0] + rect[2] - 1, rect[1] + rect[3] - 1]  # x1,y1,x2,y2
            #
            pos, s = crop_like_SiamFCx(bbox, exemplar_size=127, context_amount=0.5, search_size=511)

            img_new = crop_like_SiamFCx1(im, bbox, exemplar_size=127, context_amount=0.5, search_size=511, padding=(0, 0, 0))

            # print(img_new.shape)
            # print(type(img_new))

            mapping_bbox = pos_s_2_bbox(pos, s)

            mapping = crop_hwc(mapping_bbox, out_sz = 511)

            ann_keypoints = ann['keypoints']

            keypoints = kp_conversion(ann_keypoints, mapping)

            # x = []
            # y = []
            # for i in range(17):
            #     x.append(keypoints[i * 3 + 0])
            #     y.append(keypoints[i * 3 + 1])
            #
            # plt.xlim(0,511)
            # plt.ylim(0,511)
            # ax = plt.gca()
            # ax.xaxis.set_ticks_position = 'top'
            # ax.invert_yaxis()
            # plt.scatter(x, y, marker='o', s=40)
            # plt.show()
            # img_new = np.transpose(img_new, (0,1,2)).astype(np.int16)
            # print(img_new.shape)
            # for i in range(17):
            #     x = int(keypoints[i * 3 + 0])
            #     y = int(keypoints[i * 3 + 1])
            #     cv2.circle(img_new, (x, y), 3, (0, 0, 213), -1)
            # plt.figure(figsize=(20, 18))
            # plt.imshow(img_new)
            # plt.show()

            dataset[crop_base_path]['{:02d}'.format(track_id)] = {'000000': bbox, 'kp_000000': keypoints}
            # dataset[crop_base_path]['{:02d}'.format(track_id)] = {'000000': bbox}
            # del_base_path = crop_base_path
            # print(dataset[crop_base_path])

        if crop_base_path in dataset.keys():
            if len(dataset[crop_base_path]) == 0:
                # print('empty')
                del dataset[crop_base_path]


    print('save json (dataset), please wait 20 seconds~')
    json.dump(dataset, open('{}_pose_siamfc.json'.format(data_subset), 'w'), indent=4, sort_keys=True)
    json.dump(dataset, open('{}_pose_siamfc.json'.format(data_subset), 'w'), indent=4, sort_keys=True)
    print('done!')

