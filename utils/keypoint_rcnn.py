# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
"""Construct minibatches for Mask R-CNN training when keypoints are enabled.
Handles the minibatch blobs that are specific to training Mask R-CNN for
keypoint detection. Other blobs that are generic to RPN or Fast/er R-CNN are
handled by their respecitive roi_data modules.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np


def heatmaps_to_keypoints(maps, rois):
    """Extract predicted keypoint locations from heatmaps. Output has shape
    (#rois, 4, #keypoints) with the 4 rows corresponding to (x, y, logit, prob)
    for each keypoint.
    """
    # This function converts a discrete image coordinate in a HEATMAP_SIZE x
    # HEATMAP_SIZE image to a continuous keypoint coordinate. We maintain
    # consistency with keypoints_to_heatmap_labels by using the conversion from
    # Heckbert 1990: c = d + 0.5, where d is a discrete coordinate and c is a
    # continuous coordinate.
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]

    widths = rois[:, 2] - rois[:, 0]
    heights = rois[:, 3] - rois[:, 1]
    widths = np.maximum(widths, 1)
    heights = np.maximum(heights, 1)
    widths_ceil = np.ceil(widths)
    heights_ceil = np.ceil(heights)

    # NCHW to NHWC for use with OpenCV
    maps = np.transpose(maps, [0, 2, 3, 1])
    min_size = cfg.KRCNN.INFERENCE_MIN_SIZE
    xy_preds = np.zeros(
        (len(rois), 4, cfg.KRCNN.NUM_KEYPOINTS), dtype=np.float32)
    for i in range(len(rois)):
        if min_size > 0:
            roi_map_width = int(np.maximum(widths_ceil[i], min_size))
            roi_map_height = int(np.maximum(heights_ceil[i], min_size))
        else:
            roi_map_width = widths_ceil[i]
            roi_map_height = heights_ceil[i]
        width_correction = widths[i] / roi_map_width
        height_correction = heights[i] / roi_map_height
        roi_map = cv2.resize(
            maps[i], (roi_map_width, roi_map_height),
            interpolation=cv2.INTER_CUBIC)
        # Bring back to CHW
        roi_map = np.transpose(roi_map, [2, 0, 1])
        roi_map_probs = scores_to_probs(roi_map.copy())
        w = roi_map.shape[2]
        for k in range(cfg.KRCNN.NUM_KEYPOINTS):
            pos = roi_map[k, :, :].argmax()
            x_int = pos % w
            y_int = (pos - x_int) // w
            assert (roi_map_probs[k, y_int, x_int] ==
                    roi_map_probs[k, :, :].max())
            x = (x_int + 0.5) * width_correction
            y = (y_int + 0.5) * height_correction
            xy_preds[i, 0, k] = x + offset_x[i]
            xy_preds[i, 1, k] = y + offset_y[i]
            xy_preds[i, 2, k] = roi_map[k, y_int, x_int]
            xy_preds[i, 3, k] = roi_map_probs[k, y_int, x_int]

    return xy_preds


def keypoints_to_heatmap_labels(keypoints, rois, num_kps=17, heatmap_size=56):
    """Encode keypoint location in the target heatmap for use in
    SoftmaxWithLoss.
    """
    # Maps keypoints from the half-open interval [x1, x2) on continuous image
    # coordinates to the closed interval [0, HEATMAP_SIZE - 1] on discrete image
    # coordinates. We use the continuous <-> discrete conversion from Heckbert
    # 1990 ("What is the coordinate of a pixel?"): d = floor(c) and c = d + 0.5,
    # where d is a discrete coordinate and c is a continuous coordinate.
    assert keypoints.shape[2] == num_kps

    shape = (len(rois), num_kps)
    heatmaps = np.zeros(shape, dtype=np.float32)
    weights = np.zeros(shape, dtype=np.int32)

    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    scale_x = heatmap_size / (rois[:, 2] - rois[:, 0])
    scale_y = heatmap_size / (rois[:, 3] - rois[:, 1])

    for kp in range(keypoints.shape[2]):
        vis = keypoints[:, 2, kp] > 0
        x = keypoints[:, 0, kp].astype(np.float32)
        y = keypoints[:, 1, kp].astype(np.float32)
        # Since we use floor below, if a keypoint is exactly on the roi's right
        # or bottom boundary, we shift it in by eps (conceptually) to keep it in
        # the ground truth heatmap.
        x_boundary_inds = np.where(x == rois[:, 2])[0]
        y_boundary_inds = np.where(y == rois[:, 3])[0]
        x = (x - offset_x) * scale_x
        x = np.floor(x)
        if len(x_boundary_inds) > 0:
            x[x_boundary_inds] = heatmap_size - 1

        y = (y - offset_y) * scale_y
        y = np.floor(y)
        if len(y_boundary_inds) > 0:
            y[y_boundary_inds] = heatmap_size - 1

        valid_loc = np.logical_and(
            np.logical_and(x >= 0, y >= 0),
            np.logical_and(
                x < heatmap_size, y < heatmap_size))

        valid = np.logical_and(valid_loc, vis)
        valid = valid.astype(np.int32)

        lin_ind = y * heatmap_size + x
        heatmaps[:, kp] = lin_ind * valid
        weights[:, kp] = valid

    return heatmaps, weights


def add_keypoint_rcnn_gts(gt_keypoints, boxes, batch_idx, num_kps=17, img_size=255):
    # gt_keypoints: [bs, 3, num_keypoints] bcause only one person per image
    # boxes: [num_rois, 4]
    # batch_idx: [num_rois]
    gt_keypoints = gt_keypoints.detach().cpu().numpy()
    boxes = boxes*img_size
    boxes = boxes.detach().cpu().numpy()
    batch_idx = batch_idx.int().detach().cpu().numpy()
    gt_keypoints = gt_keypoints[batch_idx]
    assert gt_keypoints.shape[0] == boxes.shape[0]
    # print('gt_keypoints shape: ', gt_keypoints.shape)
    within_box = _within_box(gt_keypoints, boxes)
    # print('within_box shape: ', within_box.shape)
    # vis_kp = gt_keypoints[:, 2, :] > 0
    # is_visible = np.sum(np.logical_and(vis_kp, within_box), axis=1)
    # kp_fg_inds = np.where(is_visible > 0)[0]
    # print('kp_fg_inds shape: ', kp_fg_inds.shape)

    sampled_fg_rois = boxes # boxes[kp_fg_inds]

    # kp_fg_rois_per_this_image = np.minimum(fg_rois_per_image, kp_fg_inds.size)
    # if kp_fg_inds.size > kp_fg_rois_per_this_image:
    #     kp_fg_inds = np.random.choice(
    #         kp_fg_inds, size=kp_fg_rois_per_this_image, replace=False)

    num_keypoints = gt_keypoints.shape[2]
    sampled_keypoints = -np.ones(
        (len(sampled_fg_rois), gt_keypoints.shape[1], num_keypoints),
        dtype=gt_keypoints.dtype)

    for ii in range(len(sampled_fg_rois)):
        sampled_keypoints[ii, :, :] = gt_keypoints[ii, :, :]
        assert np.sum(sampled_keypoints[ii, 2, :]) > 0

    heats, weights = keypoints_to_heatmap_labels(
        sampled_keypoints, sampled_fg_rois)

    # heats = heats.reshape(shape)
    # weights = weights.reshape(shape)

    return sampled_fg_rois, heats.astype(np.int32, copy=False), weights

# def finalize_keypoint_minibatch(blobs, valid):
#     """Finalize the minibatch after blobs for all minibatch images have been
#     collated.
#     """
#     min_count = cfg.KRCNN.MIN_KEYPOINT_COUNT_FOR_VALID_MINIBATCH
#     num_visible_keypoints = np.sum(blobs['keypoint_weights'])
#     valid = (valid and len(blobs['keypoint_weights']) > 0
#              and num_visible_keypoints > min_count)
#     # Normalizer to use if cfg.KRCNN.NORMALIZE_BY_VISIBLE_KEYPOINTS is False.
#     # See modeling.model_builder.add_keypoint_losses
#     norm = num_visible_keypoints / (
#         cfg.TRAIN.IMS_PER_BATCH * cfg.TRAIN.BATCH_SIZE_PER_IM * cfg.TRAIN.
#         FG_FRACTION * cfg.KRCNN.NUM_KEYPOINTS)
#     blobs['keypoint_loss_normalizer'] = np.array(norm, dtype=np.float32)
#     return valid


def _within_box(points, boxes):
    """Validate which keypoints are contained inside a given box.

    points: Nx2xK
    boxes: Nx4
    output: NxK
    """
    x_within = np.logical_and(
        points[:, 0, :] >= np.expand_dims(boxes[:, 0], axis=1),
        points[:, 0, :] <= np.expand_dims(boxes[:, 2], axis=1))
    y_within = np.logical_and(
        points[:, 1, :] >= np.expand_dims(boxes[:, 1], axis=1),
        points[:, 1, :] <= np.expand_dims(boxes[:, 3], axis=1))
    return np.logical_and(x_within, y_within)
