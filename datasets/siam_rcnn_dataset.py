# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
from __future__ import division
from torch.utils.data import Dataset
import numpy as np
import json
import random
import logging
from os.path import join
from utils.bbox_helper import *
from utils.anchors import Anchors
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
import math
import sys
pyv = sys.version[0]
import cv2
if pyv[0] == '3':
    cv2.ocl.setUseOpenCL(False)

logger = logging.getLogger('global')


sample_random = random.Random()
sample_random.seed(123456)


class SubDataSet(object):
    def __init__(self, cfg):
        for string in ['root', 'anno']:
            if string not in cfg:
                raise Exception('SubDataSet need "{}"'.format(string))

        with open(cfg['anno']) as fin:
            logger.info("loading " + cfg['anno'])
            self.labels = self.filter_zero(json.load(fin), cfg)

            def isint(x):
                try:
                    int(x)
                    return True
                except:
                    return False

            # add frames args into labels
            to_del = []
            for video in self.labels:
                for track in self.labels[video]:
                        frames = self.labels[video][track]
                        frames = list(map(int, filter(lambda x: isint(x), frames.keys())))
                        frames.sort()
                        self.labels[video][track]['frames'] = frames
                        if len(frames) <= 0:
                            logger.info("warning {}/{} has no frames.".format(video, track))
                            to_del.append((video, track))

            # delete tracks with no frames
            for video, track in to_del:
                del self.labels[video][track]

            # delete videos with no valid track
            to_del = []
            for video in self.labels:
                if len(self.labels[video]) <= 0:
                    logger.info("warning {} has no tracks".format(video))
                    to_del.append(video)

            for video in to_del:
                del self.labels[video]

            self.videos = list(self.labels.keys())

            logger.info(cfg['anno'] + " loaded.")

        # default args
        self.root = "/"
        self.start = 0
        self.num = len(self.labels)
        self.num_use = self.num
        self.frame_range = 100
        self.mark = "vid"
        self.path_format = "{}.{}.{}.jpg"
        self.mask_format = "{}.{}.m.png"

        self.pick = []

        # input args
        self.__dict__.update(cfg)

        self.has_mask = self.mark in ['coco', 'ytb_vos'] and 'mask' in cfg
        # print('has mask: ', self.has_mask)

        self.num_use = int(self.num_use)

        # shuffle
        self.shuffle()

    def filter_zero(self, anno, cfg):
        name = cfg.get('mark', '')

        out = {}
        tot = 0
        new = 0
        zero = 0

        for video, tracks in anno.items():
            new_tracks = {}
            for trk, frames in tracks.items():
                new_frames = {}
                for frm, bbox in frames.items():
                    tot += 1
                    if 'kp' in frm:
                        new_frames[frm] = bbox
                    else:
                        if len(bbox) == 4:
                            x1, y1, x2, y2 = bbox
                            w, h = x2 - x1, y2 - y1
                        else:
                            w, h = bbox
                        if w == 0 or h == 0:
                            logger.info('Error, {name} {video} {trk} {bbox}'.format(**locals()))
                            zero += 1
                            continue
                        new += 1
                        new_frames[frm] = bbox

                if len(new_frames) > 0:
                    new_tracks[trk] = new_frames

            if len(new_tracks) > 0:
                out[video] = new_tracks

        return out

    def log(self):
        logger.info('SubDataSet {name} start-index {start} select [{select}/{num}] path {format}'.format(
            name=self.mark, start=self.start, select=self.num_use, num=self.num, format=self.path_format
        ))

    def shuffle(self):
        lists = list(range(self.start, self.start + self.num))

        m = 0
        pick = []
        while m < self.num_use:
            sample_random.shuffle(lists)
            pick += lists
            m += self.num

        self.pick = pick[:self.num_use]
        return self.pick

    def get_image_anno(self, video, track, frame, out='keypoints'):
        frame = "{:06d}".format(frame)
        image_path = join(self.root, video, self.path_format.format(frame, track, 'x'))
        image_anno = self.labels[video][track][frame]
        if out == 'keypoints':
            output = self.labels[video][track]['kp_'+frame]
        elif out == 'mask':
            output = join(self.root, video, self.mask_format.format(frame, track))

        return image_path, image_anno, output


    def get_positive_pair(self, index):
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = random.choice(list(video.keys()))
        track_info = video[track]

        frames = track_info['frames']

        if 'hard' not in track_info:
            template_frame = random.randint(0, len(frames)-1)

            left = max(template_frame - self.frame_range, 0)
            right = min(template_frame + self.frame_range, len(frames)-1) + 1
            search_range = frames[left:right]
            template_frame = frames[template_frame]
            search_frame = random.choice(search_range)
        else:
            search_frame = random.choice(track_info['hard'])
            left = max(search_frame - self.frame_range, 0)
            right = min(search_frame + self.frame_range, len(frames)-1) + 1  # python [left:right+1) = [left:right]
            template_range = frames[left:right]
            template_frame = random.choice(template_range)
            search_frame = frames[search_frame]

        return self.get_image_anno(video_name, track, template_frame), \
               self.get_image_anno(video_name, track, search_frame)

    def get_random_target(self, index=-1):
        if index == -1:
            index = random.randint(0, self.num-1)
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = random.choice(list(video.keys()))
        track_info = video[track]

        frames = track_info['frames']
        frame = random.choice(frames)

        return self.get_image_anno(video_name, track, frame)


def crop_hwc(image, bbox, out_sz, padding=(0, 0, 0)):
    bbox = [float(x) for x in bbox]
    a = (out_sz-1) / (bbox[2]-bbox[0])
    b = (out_sz-1) / (bbox[3]-bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return crop


class Augmentation:
    def __init__(self, cfg):
        # default args
        self.shift = 0
        self.scale = 0
        self.blur = 0 #False
        self.resize = False
        self.rgbVar = np.array([[-0.55919361,  0.98062831, - 0.41940627],
            [1.72091413,  0.19879334, - 1.82968581],
            [4.64467907,  4.73710203, 4.88324118]], dtype=np.float32)
        self.flip = 0

        self.eig_vec = np.array([
            [0.4009, 0.7192, -0.5675],
            [-0.8140, -0.0045, -0.5808],
            [0.4203, -0.6948, -0.5836],
        ], dtype=np.float32)

        self.eig_val = np.array([[0.2175, 0.0188, 0.0045]], np.float32)

        self.__dict__.update(cfg)

    @staticmethod
    def random():
        return random.random() * 2 - 1.0

    def blur_image(self, image):
        def rand_kernel():
            size = np.random.randn(1)
            size = int(np.round(size)) * 2 + 1
            if size < 0: return None
            if random.random() < 0.5: return None
            size = min(size, 45)
            kernel = np.zeros((size, size))
            c = int(size/2)
            wx = random.random()
            kernel[:, c] += 1. / size * wx
            kernel[c, :] += 1. / size * (1-wx)
            return kernel

        kernel = rand_kernel()

        if kernel is not None:
            image = cv2.filter2D(image, -1, kernel)
        return image

    def __call__(self, image, bbox, size, gray=False, mask=None, kp=None):
        if gray:
            grayed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = np.zeros((grayed.shape[0], grayed.shape[1], 3), np.uint8)
            image[:, :, 0] = image[:, :, 1] = image[:, :, 2] = grayed

        shape = image.shape

        crop_bbox = center2corner((shape[0]//2, shape[1]//2, size-1, size-1))

        param = {}
        if self.shift:
            param['shift'] = (Augmentation.random() * self.shift, Augmentation.random() * self.shift)

        if self.scale:
            param['scale'] = ((1.0 + Augmentation.random() * self.scale), (1.0 + Augmentation.random() * self.scale))

        crop_bbox, _ = aug_apply(Corner(*crop_bbox), param, shape)

        x1 = crop_bbox.x1
        y1 = crop_bbox.y1

        bbox = BBox(bbox.x1 - x1, bbox.y1 - y1,
                    bbox.x2 - x1, bbox.y2 - y1)
        if kp:
            kp[:, 0] = kp[:, 0] - x1
            kp[:, 1] = kp[:, 1] - y1

        if self.scale:
            scale_x, scale_y = param['scale']
            bbox = Corner(bbox.x1 / scale_x, bbox.y1 / scale_y, bbox.x2 / scale_x, bbox.y2 / scale_y)
            if kp:
                kp[:, 0] = kp[:, 0] / scale_x
                kp[:, 1] = kp[:, 1] / scale_y

        image = crop_hwc(image, crop_bbox, size)
        if not mask is None:
            mask = crop_hwc(mask, crop_bbox, size)

        offset = np.dot(self.rgbVar, np.random.randn(3, 1))
        offset = offset[::-1]  # bgr 2 rgb
        offset = offset.reshape(3)
        image = image - offset

        if self.blur > random.random():
            image = self.blur_image(image)

        if self.resize:
            imageSize = image.shape[:2]
            ratio = max(math.pow(random.random(), 0.5), 0.2)  # 25 ~ 255
            rand_size = (int(round(ratio*imageSize[0])), int(round(ratio*imageSize[1])))
            image = cv2.resize(image, rand_size)
            image = cv2.resize(image, tuple(imageSize))

        if self.flip and self.flip > Augmentation.random():
            image = cv2.flip(image, 1)
            width = image.shape[1]
            mask = cv2.flip(mask, 1)
            bbox = Corner(width - 1 - bbox.x2, bbox.y1, width - 1 - bbox.x1, bbox.y2)
            if kp:
                kp[:, 0] = width - 1 - kp[:, 0]

        return image, bbox, mask, kp


class AnchorTargetLayer:
    def __init__(self, cfg):
        self.thr_high = 0.6
        self.thr_low = 0.3
        self.negative = 16
        self.rpn_batch = 64
        self.positive = 16

        self.__dict__.update(cfg)

    def __call__(self, anchor, target, size, neg=False, need_iou=False):
        anchor_num = anchor.anchors.shape[0]

        cls = np.zeros((anchor_num, size, size), dtype=np.int64)
        cls[...] = -1  # -1 ignore 0 negative 1 positive
        delta = np.zeros((4, anchor_num, size, size), dtype=np.float32)
        delta_weight = np.zeros((anchor_num, size, size), dtype=np.float32)

        def select(position, keep_num=16):
            num = position[0].shape[0]
            if num <= keep_num:
                return position, num
            slt = np.arange(num)
            np.random.shuffle(slt)
            slt = slt[:keep_num]
            return tuple(p[slt] for p in position), keep_num

        if neg:
            l = size // 2 - 3
            r = size // 2 + 3 + 1

            cls[:, l:r, l:r] = 0

            neg, neg_num = select(np.where(cls == 0), self.negative)
            cls[:] = -1
            cls[neg] = 0

            if not need_iou:
                return cls, delta, delta_weight
            else:
                overlap = np.zeros((anchor_num, size, size), dtype=np.float32)
                return cls, delta, delta_weight, overlap

        tcx, tcy, tw, th = corner2center(target)
        # print('tcx shape: ', tcx)


        anchor_box = anchor.all_anchors[0]
        anchor_center = anchor.all_anchors[1]
        # print('anchor_center: ', anchor_center.shape)
        x1, y1, x2, y2 = anchor_box[0], anchor_box[1], anchor_box[2], anchor_box[3]
        cx, cy, w, h = anchor_center[0], anchor_center[1], anchor_center[2], anchor_center[3]

        # delta
        delta[0] = (tcx - cx) / w
        delta[1] = (tcy - cy) / h
        delta[2] = np.log(tw / w)
        delta[3] = np.log(th / h)

        # IoU
        overlap = IoU([x1, y1, x2, y2], target)

        pos = np.where(overlap > self.thr_high)
        neg = np.where(overlap < self.thr_low)

        pos, pos_num = select(pos, self.positive)
        neg, neg_num = select(neg, self.rpn_batch - pos_num)
        # print('pos: ', pos)

        cls[pos] = 1
        delta_weight[pos] = 1. / (pos_num + 1e-6)

        cls[neg] = 0

        if not need_iou:
            return cls, delta, delta_weight
        else:
            return cls, delta, delta_weight, overlap


class AnchorTargetWithKPLayer:
    def __init__(self, cfg):
        self.thr_high = 0.6
        self.thr_low = 0.3
        self.negative = 16
        self.rpn_batch = 64
        self.positive = 16

        self.__dict__.update(cfg)

    def __call__(self, anchor, target, kp, size, neg=False, need_iou=False):
        anchor_num = anchor.anchors.shape[0]
        # kp shape [17, 3]
        cls = np.zeros((anchor_num, size, size), dtype=np.int64)
        cls[...] = -1  # -1 ignore 0 negative 1 positive
        delta = np.zeros((4, anchor_num, size, size), dtype=np.float32)
        delta_weight = np.zeros((anchor_num, size, size), dtype=np.float32)
        kp_delta = np.zeros((3, 17, size, size), dtype=np.float32)

        def select(position, keep_num=16):
            num = position[0].shape[0]
            if num <= keep_num:
                return position, num
            slt = np.arange(num)
            np.random.shuffle(slt)
            slt = slt[:keep_num]
            return tuple(p[slt] for p in position), keep_num

        if neg:
            l = size // 2 - 3
            r = size // 2 + 3 + 1

            cls[:, l:r, l:r] = 0

            neg, neg_num = select(np.where(cls == 0), self.negative)
            cls[:] = -1
            cls[neg] = 0

            if not need_iou:
                return cls, delta, kp_delta, delta_weight
            else:
                overlap = np.zeros((anchor_num, size, size), dtype=np.float32)
                return cls, delta, kp_delta, delta_weight, overlap

        tcx, tcy, tw, th = corner2center(target)


        anchor_box = anchor.all_anchors[0]
        anchor_center = anchor.all_anchors[1]
        x1, y1, x2, y2 = anchor_box[0], anchor_box[1], anchor_box[2], anchor_box[3]
        cx, cy, w, h = anchor_center[0], anchor_center[1], anchor_center[2], anchor_center[3]
        # cx shape: [anchor_num, size, size] -> [size, size]
        # kp shape: [17, 3]
        # kp delta shape: [17*2, size, size]
        # kp_delta_x target shape: [17, size, size]
        # kp_x [17, 1, 1] and cx_kp [1, size, size]

        cx_kp = np.expand_dims(cx[0, :, :], 0)
        cy_kp = np.expand_dims(cx[0, :, :], 0)
        kp_x = np.expand_dims(np.expand_dims(kp[:, 0], -1), -1)
        kp_y = np.expand_dims(np.expand_dims(kp[:, 1], -1), -1)
        kp_vis = np.expand_dims(np.expand_dims(kp[:, 2], -1), -1)

        kp_vis = np.repeat(kp_vis, size, axis=1)
        kp_vis = np.repeat(kp_vis, size, axis=2)

        kp_delta_x = (kp_x - cx_kp)  # / w[0, ...]  # (17, size, size)
        kp_delta_y = (kp_y - cy_kp)  # / h[0, ...]
        kp_delta = np.stack([kp_delta_x, kp_delta_y, kp_vis], axis=0)  # (3, 17, size, size)
        kp_delta = kp_delta.astype(np.float32)

        # delta
        delta[0] = (tcx - cx) / w
        delta[1] = (tcy - cy) / h
        delta[2] = np.log(tw / w)
        delta[3] = np.log(th / h)

        # IoU
        overlap = IoU([x1, y1, x2, y2], target)

        pos = np.where(overlap > self.thr_high)
        neg = np.where(overlap < self.thr_low)

        pos, pos_num = select(pos, self.positive)
        neg, neg_num = select(neg, self.rpn_batch - pos_num)

        cls[pos] = 1
        delta_weight[pos] = 1. / (pos_num + 1e-6)

        cls[neg] = 0

        if not need_iou:
            return cls, delta, kp_delta, delta_weight
        else:
            return cls, delta, kp_delta, delta_weight, overlap


class DataSets(Dataset):
    def __init__(self, cfg, anchor_cfg):
        super(DataSets, self).__init__()
        global logger
        logger = logging.getLogger('global')

        # anchors
        self.anchors = Anchors(anchor_cfg)

        # size
        self.template_size = 127
        self.origin_size = 127
        self.search_size = 255
        self.heatmap_size = (255, 255)
        self.image_size = 255
        self.size = 17
        self.sigma = 4
        self.base_size = 0
        self.crop_size = 0
        self.target_type = 'gaussian'
        self.single_heatmap = False
        self.output_res = 56
        self.num_joints = 17                                                # added
        self.mse_loss = False
        self.hm_gauss = 5


        if 'template_size' in cfg:
            self.template_size = cfg['template_size']
        if 'origin_size' in cfg:
            self.origin_size = cfg['origin_size']
        if 'search_size' in cfg:
            self.search_size = cfg['search_size']
        if 'base_size' in cfg:
            self.base_size = cfg['base_size']
        if 'size' in cfg:
            self.size = cfg['size']
        if 'single_heatmap' in cfg:
            self.single_heatmap = cfg['single_heatmap']

        if (self.search_size - self.template_size) / self.anchors.stride + 1 + self.base_size != self.size:
            raise Exception("size not match!")  # TODO: calculate size online
        if 'crop_size' in cfg:
            self.crop_size = cfg['crop_size']
        self.template_small = False
        if 'template_small' in cfg and cfg['template_small']:
            self.template_small = True

        self.anchors.generate_all_anchors(im_c=self.search_size//2, size=self.size)

        if 'anchor_target' not in cfg:
            cfg['anchor_target'] = {}
        if 'kp_anchor' not in anchor_cfg:
            self.anchor_target = AnchorTargetLayer(cfg['anchor_target'])
            self.kp_anchor = False
        else:
            self.anchor_target = AnchorTargetWithKPLayer(cfg['anchor_target'])
            self.kp_anchor = True

        # data sets
        if 'datasets' not in cfg:
            raise(Exception('DataSet need "{}"'.format('datasets')))

        self.all_data = []
        start = 0
        self.num = 0
        for name in cfg['datasets']:
            dataset = cfg['datasets'][name]
            dataset['mark'] = name
            dataset['start'] = start

            dataset = SubDataSet(dataset)
            dataset.log()
            self.all_data.append(dataset)

            start += dataset.num  # real video number
            self.num += dataset.num_use  # the number used for subset shuffle

        # data augmentation
        aug_cfg = cfg['augmentation']
        self.template_aug = Augmentation(aug_cfg['template'])
        self.search_aug = Augmentation(aug_cfg['search'])
        self.gray = aug_cfg['gray']
        self.neg = aug_cfg['neg']
        self.inner_neg = 0 if 'inner_neg' not in aug_cfg else aug_cfg['inner_neg']

        self.pick = None  # list to save id for each img
        if 'num' in cfg:  # number used in training for all dataset
            self.num = int(cfg['num'])
        self.shuffle()

        self.infos = {
                'template': self.template_size,
                'search': self.search_size,
                'template_small': self.template_small,
                'gray': self.gray,
                'neg': self.neg,
                'inner_neg': self.inner_neg,
                'crop_size': self.crop_size,
                'anchor_target': self.anchor_target.__dict__,
                'num': self.num
                }
        logger.info('dataset informations: \n{}'.format(json.dumps(self.infos, indent=4)))

    def imread(self, path):
        img = cv2.imread(path)

        if self.origin_size == self.template_size:
            return img, 1.0

        def map_size(exe, size):
            return int(round(((exe + 1) / (self.origin_size + 1) * (size+1) - 1)))

        nsize = map_size(self.template_size, img.shape[1])

        img = cv2.resize(img, (nsize, nsize))

        return img, nsize / img.shape[1]

    def shuffle(self):
        pick = []
        m = 0
        while m < self.num:
            p = []
            for subset in self.all_data:
                sub_p = subset.shuffle()
                p += sub_p

            sample_random.shuffle(p)

            pick += p
            m = len(pick)
        self.pick = pick
        logger.info("shuffle done!")
        logger.info("dataset length {}".format(self.num))

    def __len__(self):
        return self.num

    def find_dataset(self, index):
        for dataset in self.all_data:
            if dataset.start + dataset.num > index:
                return dataset, index - dataset.start

    def __getitem__(self, index, debug=False):
        index = self.pick[index]
        dataset, index = self.find_dataset(index)

        gray = self.gray and self.gray > random.random()
        neg = self.neg and self.neg > random.random()

        if neg:
            template = dataset.get_random_target(index)
            if self.inner_neg and self.inner_neg > random.random():
                search = dataset.get_random_target()
            else:
                search = random.choice(self.all_data).get_random_target()
        else:
            template, search = dataset.get_positive_pair(index)

        def center_crop(img, size):
            shape = img.shape[1]
            if shape == size: return img
            c = shape // 2
            l = c - size // 2
            r = c + size // 2 + 1
            return img[l:r, l:r]

        template_image, scale_z = self.imread(template[0])

        if self.template_small:
            template_image = center_crop(template_image, self.template_size)

        search_image, scale_x = self.imread(search[0])

        if dataset.has_mask:
            if not neg:
                search_mask = (cv2.imread(search[2], 0) > 0).astype(np.float32)
            else:
                search_mask = np.zeros(search_image.shape[:2], dtype=np.float32)
        else:
            if not neg:
                search_kp = np.array(search[2], dtype=np.float32)
            else:
                search_kp = np.zeros(51, dtype=np.float32)

        if self.crop_size > 0:
            search_image = center_crop(search_image, self.crop_size)

        def toBBox(image, shape, context_amount=0):
            imh, imw = image.shape[:2]
            if len(shape) == 4:
                w, h = shape[2]-shape[0], shape[3]-shape[1]
            else:
                w, h = shape

            exemplar_size = self.template_size  # 127
            wc_z = w + context_amount * (w+h)
            hc_z = h + context_amount * (w+h)
            s_z = np.sqrt(wc_z * hc_z)
            scale_z = exemplar_size / s_z
            w = w*scale_z
            h = h*scale_z
            cx, cy = imw//2, imh//2
            bbox = center2corner(Center(cx, cy, w, h))
            return bbox

        def toKP(image, shape, kp, context_amount=0):
            imh, imw = image.shape[:2]
            output_kp = np.zeros_like(kp)

            if len(shape) == 4:
                w, h = shape[2]-shape[0], shape[3]-shape[1]
            else:
                w, h = shape

            bbox_center_x = (shape[2]+shape[0])/2
            bbox_center_y = (shape[3]+shape[1])/2
            kp_x = kp[:, 0]
            kp_y = kp[:, 1]
            rel_kp_x = kp_x - bbox_center_x
            rel_kp_y = kp_y - bbox_center_y

            exemplar_size = self.template_size  # 127
            wc_z = w + context_amount * (w+h)
            hc_z = h + context_amount * (w+h)
            s_z = np.sqrt(wc_z * hc_z)
            scale_z = exemplar_size / s_z
            rel_kp_x = rel_kp_x*scale_z
            rel_kp_y = rel_kp_y*scale_z
            cx, cy = imw//2, imh//2

            output_kp[:, 0] = rel_kp_x + cx
            output_kp[:, 1] = rel_kp_y + cy
            output_kp[:, 2] = kp[:, 2]
            return output_kp

        template_box = toBBox(template_image, template[1])
        search_box = toBBox(search_image, search[1])
        search_kp = toKP(search_image, search[1], search_kp)
        # bbox = search_box
        template, _, _, _ = self.template_aug(template_image, template_box, self.template_size, gray=gray)
        search, bbox, mask, kp = self.search_aug(search_image, search_box, self.search_size, gray=gray, kp=search_kp)

        def draw(image, box, name):
            image = image.copy()
            x1, y1, x2, y2 = map(lambda x: int(round(x)), box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0))
            cv2.imwrite(name, image)

        def crop_hwc(bbox, out_sz=255):
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
            a = (out_sz - 1) / (bbox[2] - bbox[0])
            b = (out_sz - 1) / (bbox[3] - bbox[1])
            c = -a * bbox[0]
            d = -b * bbox[1]
            mapping = np.array([[a, 0, c],
                                [0, b, d]]).astype(np.float)
            crop = cv2.warpAffine(image, mapping, (out_sz, out_sz))
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

        def kp_conversion(KeyPoints, matrix):

            key_points = []
            kps_conversion = []
            skeleton = [0, 0]
            Skeleton = []

            for i in range(0, int(len(KeyPoints) / 3)):
                skeleton[0] = KeyPoints[i * 3 + 0]
                skeleton[1] = KeyPoints[i * 3 + 1]
                Skeleton.append(skeleton[:])
                lis = Skeleton[i]
                lis.append(1)
                key_points.append(lis)

            key_points = np.array(key_points)

            for i in range(0, int(len(KeyPoints) / 3)):
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

        if debug:
            draw(template_image, template_box, "debug/{:06d}_ot.jpg".format(index))
            draw(search_image, search_box, "debug/{:06d}_os.jpg".format(index))
            draw(template, _, "debug/{:06d}_t.jpg".format(index))
            draw(search, bbox, "debug/{:06d}_s.jpg".format(index))

        cls, delta, delta_weight = self.anchor_target(self.anchors, bbox, self.size, neg)
        if not dataset.has_mask:
            pos, s = crop_like_SiamFCx(search_box, exemplar_size=127, context_amount=0.5, search_size=255)
            mapping_bbox = pos_s_2_bbox(pos, s)

            mapping = crop_hwc(mapping_bbox, out_sz=255)

            keypoints = kp_conversion(search_kp.tolist(), mapping)

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float)
            for ipt in range(self.num_joints):
                joints_3d[ipt, 0] = keypoints[ipt * 3 + 0]
                joints_3d[ipt, 1] = keypoints[ipt * 3 + 1]
                joints_3d[ipt, 2] = keypoints[ipt * 3 + 2]
                t_vis = search_kp[ipt * 3 + 2]
                if t_vis > 1:
                    t_vis = 1
                joints_3d_vis[ipt, 0] = t_vis
                joints_3d_vis[ipt, 1] = t_vis
                joints_3d_vis[ipt, 2] = 0

            img = search.copy()
            # joints_3d = joints_3d / 255

            if not neg:
                kp_weight = cls.max(axis=0, keepdims=True)
            else:
                kp_weight = np.zeros([1, cls.shape[1], cls.shape[2]], dtype=np.float32)


            # now process the ct part
            c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
            s = max(img.shape[0], img.shape[1]) * 1.0
            rot = 0

            output_res = self.output_res
            num_joints = self.num_joints
            trans_output_rot = get_affine_transform(c, s, rot, [output_res, output_res])
            trans_output = get_affine_transform(c, s, 0, [output_res, output_res])

            ind = np.zeros(1, dtype=np.int64)
            # hm_hp = np.zeros((num_joints, output_res, output_res), dtype=np.float32)
            hm_hp = np.zeros((num_joints, output_res, output_res), dtype=np.float32)
            kps = np.zeros(num_joints * 2, dtype=np.float32)
            kps_mask = np.zeros((self.num_joints * 2), dtype=np.uint8)
            hp_offset = np.zeros((num_joints, 2), dtype=np.float32)
            hp_ind = np.zeros(num_joints, dtype=np.int64)
            hp_mask = np.zeros(num_joints, dtype=np.int64)


            draw_gaussian = draw_msra_gaussian if self.mse_loss else \
                        draw_umich_gaussian

            pts = joints_3d.copy()
            bbox = np.array(bbox, np.float32)
            bbox_reg = np.array(bbox, np.float32)
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox = np.clip(bbox, 0, output_res - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if (h > 0 and w > 0):
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
            hp_radius = gaussian_radius((math.ceil(h)*2.3, math.ceil(w)*2.3))
            hp_radius = self.hm_gauss \
                        if self.mse_loss else max(0, int(hp_radius))
            ind[0] = ct_int[1] * output_res + ct_int[0]
            for j in range(num_joints):
                if pts[j, 2] > 0:
                    pts[j, :2] = affine_transform(pts[j, :2], trans_output_rot)
                    if pts[j, 0] >= 0 and pts[j, 0] < output_res and \
                       pts[j, 1] >= 0 and pts[j, 1] < output_res:
                        kps[j * 2: j * 2 + 2] = pts[j, :2] - ct_int
                        kps_mask[j * 2: j * 2 + 2] = 1
                        pt_int = pts[j, :2].astype(np.int32)
                        # print('ct_int: ', ct_int)
                        # print('pt_int: ', pt_int)
                        hp_offset[j] = pts[j, :2] - pt_int
                        hp_ind[j] = pt_int[1] * output_res + pt_int[0]
                        hp_mask[j] = 1

                        draw_gaussian(hm_hp[j], pt_int, hp_radius)
                        # pt_ori = joints_3d[j, :2].astype(np.int32)
                        # draw_gaussian(hm_hp[j], pt_ori, hp_radius)

            ret = {'hps': kps, 'hm_hp': hm_hp, 'hp_mask': hp_mask}
            # print('kps: ', ret['hps'])
            ret.update({'hp_offset': hp_offset, 'hp_ind': hp_ind, 'hps_mask': kps_mask, 'ind': ind})

        # print('hp_offset: ', hp_offset)
        joints_3d_out = joints_3d.transpose(1, 0)

        template, search = map(lambda x: np.transpose(x, (2, 0, 1)).astype(np.float32), [template, search])
        return template, search, cls, delta, \
          delta_weight, bbox_reg, \
          np.array(kp_weight, np.float32), ret, joints_3d_out
