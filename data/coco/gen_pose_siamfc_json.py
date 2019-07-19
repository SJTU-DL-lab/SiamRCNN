# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
from pycocotools.coco import COCO
from os.path import join
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

dataDir = '.'
search_size = 511
exemplar_size=127
context_amount=0.5
num_keypoints=17

def crop_hwc(image, bbox, out_sz, padding=(0, 0, 0)):
    a = (out_sz-1) / (bbox[2]-bbox[0])
    b = (out_sz-1) / (bbox[3]-bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz),
                          borderMode=cv2.BORDER_CONSTANT, borderValue=padding, flags=cv2.INTER_NEAREST)
    return crop


def pos_s_2_bbox(pos, s):
    return [pos[0]-s/2, pos[1]-s/2, pos[0]+s/2, pos[1]+s/2]


def crop_like_SiamFCx(image, bbox, exemplar_size=127, context_amount=0.5, search_size=255, padding=(0, 0, 0)):
    target_pos = [(bbox[2]+bbox[0])/2., (bbox[3]+bbox[1])/2.]
    target_size = [bbox[2]-bbox[0]+1, bbox[3]-bbox[1]+1]
    wc_z = target_size[1] + context_amount * sum(target_size)
    hc_z = target_size[0] + context_amount * sum(target_size)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = exemplar_size / s_z
    d_search = (search_size - exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    x = crop_hwc(image, pos_s_2_bbox(target_pos, s_x), search_size, padding)
    return x

def keypoints2arr(kps, h, w):
    keypoints_img = np.ones((h,w,3), dtype = float) + 99
    for i in range(int(len(kps)/3)):
        if kps[i * 3 + 2] !=0:
            keypoints_img[kps[i * 3 + 1], kps[i * 3 + 0], 0] = i
            keypoints_img[kps[i * 3 + 1], kps[i * 3 + 0], 1] = kps[i * 3 + 2]
    return keypoints_img

def load_kps(arr, num):
    keypoints = []
    for i in range(num):
        index = np.argwhere(arr[:, :, 0] == i).tolist()
        if index != []:
            keypoints.append(index[0][1])
            keypoints.append(index[0][0])
            keypoints.append(arr[index[0][0], index[0][1], 1])
        else:
            keypoints.append(0)
            keypoints.append(0)
            keypoints.append(0)
    return keypoints

# for data_subset in ['val2017', 'train2017']:
for data_subset in ['val2017']:
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

        h = im.shape[0]
        w = im.shape[1]

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

            kps_arr = keypoints2arr(ann['keypoints'], h, w)

            # print(kps_arr[:,:,0])

            avg_chans = np.mean(kps_arr, axis=(0,1))
            yavg_chans = np.mean(im, axis=(0,1))

            x = crop_like_SiamFCx(kps_arr, bbox, exemplar_size=exemplar_size, context_amount=context_amount,
                              search_size=search_size, padding=avg_chans)
            y = crop_like_SiamFCx(im, bbox, exemplar_size=exemplar_size, context_amount=context_amount,
                              search_size=search_size, padding=yavg_chans)
            print(x.shape)
            print('{}/{}'.format(set_img_base_path, img['file_name']))
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    if x[i, j, 0] < 100:
                        print(x[i, j, 0])
                        x[i, j, 0] = 255
                    else:
                        x[i, j, 0] = 0
            # plt.figure(num='astronaut', figsize=(8, 8))
            plt.subplot(2, 2, 1)
            plt.title('origin image')
            plt.imshow(x[:,:,0])

            plt.subplot(2, 2, 2)
            plt.title('new image')
            plt.imshow(y)

            plt.subplot(2, 2, 3)
            plt.title('old image')
            plt.imshow(im)

            plt.show()


            keypoints = load_kps(x, num_keypoints)
            # print(bbox)
            # print(keypoints)
            dataset[crop_base_path]['{:02d}'.format(track_id)] = {'000000': bbox, 'keypoints': keypoints}

    print('save json (dataset), please wait 20 seconds~')
    json.dump(dataset, open('{}_pose_siamfcysy.json'.format(data_subset), 'w'), indent=4, sort_keys=True)
    print('done!')

