from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from utils.nms.nms_wrapper import nms
from utils.roialign.roi_align.crop_and_resize import CropAndResizeFunction

def clip_boxes(boxes, window):
    """
    boxes: [N, 4] each col is y1, x1, y2, x2
    window: [4] in the form y1, x1, y2, x2
    """
    boxes = torch.stack( \
        [boxes[:, 0].clamp(float(window[0]), float(window[2])),
         boxes[:, 1].clamp(float(window[1]), float(window[3])),
         boxes[:, 2].clamp(float(window[0]), float(window[2])),
         boxes[:, 3].clamp(float(window[1]), float(window[3]))], 1)
    return boxes

def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    return y

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim).contiguous()
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def flip_tensor(x):
    return torch.flip(x, [x.dim()-1])
    # tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    # return torch.from_numpy(tmp).to(x.device)

def flip_lr(x, flip_idx):
    tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    shape = tmp.shape
    for e in flip_idx:
        tmp[:, e[0], ...], tmp[:, e[1], ...] = \
          tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
    return torch.from_numpy(tmp.reshape(shape)).to(x.device)

def flip_lr_off(x, flip_idx):
    tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    shape = tmp.shape
    tmp = tmp.reshape(tmp.shape[0], 17, 2,
                    tmp.shape[2], tmp.shape[3])
    tmp[:, :, 0, :, :] *= -1
    for e in flip_idx:
        tmp[:, e[0], ...], tmp[:, e[1], ...] = \
          tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
    return torch.from_numpy(tmp.reshape(shape)).to(x.device)

def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, 4] where each row is x1, y1, x2, y2
    deltas: [N, 4] where each row is [dx/w, dy/h, log(dw / w), log(dh / h)]
    """
    # Convert to y, x, h, w

    height = boxes[:, 3] - boxes[:, 1]
    width = boxes[:, 2] - boxes[:, 0]
    center_y = boxes[:, 1] + 0.5 * height
    center_x = boxes[:, 0] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 1] * height
    center_x += deltas[:, 0] * width
    height *= torch.exp(deltas[:, 3])
    width *= torch.exp(deltas[:, 2])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = torch.stack([y1, x1, y2, x2], dim=1)
    return result

def proposal_layer(inputs, anchors, thresh=0.5, args=None):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinment detals to anchors.

    Inputs:
        rpn_probs: [batch, 2*anchors, height, width, (fg prob, bg prob)]
        rpn_bbox: [batch, 4*anchors, height, width]
        anchors: [?, anchors, 4, height, width, (x1, y1, x2, y2)]

    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """

    # Currently only supports batchsize 1
    # inputs[0] = inputs[0].squeeze(0)
    # inputs[1] = inputs[1].squeeze(0)

    # Box Scores, select the fg prob.
    scores = inputs[0][:, :, :, :, 1]
    scores = scores.transpose(1, 3).contiguous().view(-1)
    # print(scores > 0.3)
    pos_ix = torch.nonzero(scores > thresh).squeeze()
    # print('pos_ix: ', pos_ix.shape, pos_ix)
    # print('scores: ', scores.shape, scores[pos_ix])
    if pos_ix.size(0) == 0:
      # print('no positive ix')
      return None
      # print('positive ix')
    scores = torch.index_select(scores, 0, pos_ix)  # scores = scores[pos_ix]

    # Box deltas [batch, num_rois, 4]
    deltas = inputs[1]
    bs = deltas.size(0)
    deltas = deltas.view(-1, 4, 5, deltas.size(-2), deltas.size(-1))
    deltas = deltas.permute(0, 2, 3, 4, 1).contiguous()
    deltas = deltas.view(-1, deltas.size(4))

    boxes = torch.from_numpy(anchors).float().cuda().detach()
    boxes = boxes.expand(bs, -1, -1, -1, -1)
    boxes = boxes.transpose(0, 4).contiguous().view(-1, 4)
    assert boxes.size(0) == deltas.size(0)
    deltas = torch.index_select(deltas, 0, pos_ix)
    anchors = torch.index_select(boxes, 0, pos_ix)

    # only got 16 positive anchors, no need to remove
    # Improve performance by trimming to top anchors by score
    # and doing the rest on the smaller subset.
    # pre_nms_limit = min(6000, anchors.size()[0])
    # scores, order = scores.sort(descending=True)
    # order = order[:pre_nms_limit]
    # scores = scores[:pre_nms_limit]
    # deltas = deltas[order.data, :] # TODO: Support batch size > 1 ff.
    # anchors = anchors[order.data, :]

    # Apply deltas to anchors to get refined anchors.
    # [batch, N, (y1, x1, y2, x2)]
    boxes = apply_box_deltas(anchors, deltas)

    # Clip to image boundaries. [batch, N, (y1, x1, y2, x2)]
    size = args.img_size  # int(config['train_datasets']['search_size'])
    height, width = size, size
    window = np.array([0, 0, height, width]).astype(np.float32)
    boxes = clip_boxes(boxes, window)

    # Filter out small boxes
    # According to Xinlei Chen's paper, this reduces detection accuracy
    # for small objects, so we're skipping it.

    # Non-max suppression
    nms_threshold = args.nms_threshold  # float(config['train_datasets']['RPN_NMS'])
    # print('boxes shape: ', boxes.shape)
    # print('scores shape: ', scores.shape)
    keep = nms(torch.cat((boxes, scores.unsqueeze(1)), 1).data, nms_threshold)
    boxes = boxes[keep, :]

    # Normalize dimensions to range of 0 to 1.
    norm = Variable(torch.from_numpy(np.array([height, width, height, width])).float(), requires_grad=False)
    norm = norm.cuda()
    normalized_boxes = boxes / norm

    return normalized_boxes

def roi_align(inputs, pool_size):
    """Implements ROI Pooling on single feature map.

    Params:
    - pool_size: [height, width] of the output pooled regions. Usually [7, 7]
    - image_shape: [height, width, channels]. Shape of input image in pixels

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates.
    - Feature maps: feature map from a single feature map.
                    Each is [batch, channels, height, width]

    Output:
    Pooled regions in the shape: [num_boxes, height, width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    # Crop boxes [batch*num_boxes, (y1, x1, y2, x2)] in normalized coords
    boxes = inputs[0]
    boxes = boxes.detach()
    # boxes = boxes.unsqueeze(0)
    # boxes = boxes.transpose(0, 1).contiguous()
    # Feature Maps.
    # feature pyramid. Each is [batch, height, width, channels]
    feature_maps = inputs[1]

    # Assign each ROI to a level in the pyramid based on the ROI area.
    x1, y1, x2, y2 = boxes.chunk(4, dim=1)
    # h = y2 - y1
    # w = x2 - x1
    boxes = torch.cat([y1, x1, y2, x2], dim=1)
    # ind = torch.zeros(boxes.size()[0]).int().detach()
    ind = Variable(torch.arange(boxes.size()[0]), requires_grad=False).int() # .cuda()
    ind = ind.cuda()
    
    # print('boxes before shape: ', boxes.shape, boxes)
    # print('featmap shape: ', feature_maps.shape, feature_maps)
    # print('ind shape: ', ind.shape, ind)
    pooled_features = CropAndResizeFunction(pool_size, pool_size,
                                            0)(feature_maps, boxes, ind)

    # Equation 1 in the Feature Pyramid Networks paper. Account for
    # the fact that our coordinates are normalized here.
    # e.g. a 224x224 ROI (in pixels) maps to P4
    # image_area = Variable(torch.FloatTensor([float(image_shape[0]*image_shape[1])]), requires_grad=False)
    # if boxes.is_cuda:
    #     image_area = image_area.cuda()
    # roi_level = 4 + log2(torch.sqrt(h*w)/(224.0/torch.sqrt(image_area)))
    # roi_level = roi_level.round().int()
    # roi_level = roi_level.clamp(2,5)

    return pooled_features


def pyramid_roi_align(inputs, pool_size, image_shape):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    Params:
    - pool_size: [height, width] of the output pooled regions. Usually [7, 7]
    - image_shape: [height, width, channels]. Shape of input image in pixels

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates.
    - Feature maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, channels, height, width]

    Output:
    Pooled regions in the shape: [num_boxes, height, width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    # Currently only supports batchsize 1
    for i in range(len(inputs)):
        inputs[i] = inputs[i].squeeze(0)

    # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
    boxes = inputs[0]

    # Feature Maps. List of feature maps from different level of the
    # feature pyramid. Each is [batch, height, width, channels]
    feature_maps = inputs[1:]

    # Assign each ROI to a level in the pyramid based on the ROI area.
    y1, x1, y2, x2 = boxes.chunk(4, dim=1)
    h = y2 - y1
    w = x2 - x1

    # Equation 1 in the Feature Pyramid Networks paper. Account for
    # the fact that our coordinates are normalized here.
    # e.g. a 224x224 ROI (in pixels) maps to P4
    image_area = Variable(torch.FloatTensor([float(image_shape[0]*image_shape[1])]), requires_grad=False)
    if boxes.is_cuda:
        image_area = image_area.cuda()
    roi_level = 4 + log2(torch.sqrt(h*w)/(224.0/torch.sqrt(image_area)))
    roi_level = roi_level.round().int()
    roi_level = roi_level.clamp(2,5)


    # Loop through levels and apply ROI pooling to each. P2 to P5.
    pooled = []
    box_to_level = []
    for i, level in enumerate(range(2, 6)):
        ix  = roi_level==level
        if not ix.any():
            continue
        ix = torch.nonzero(ix)[:,0]
        level_boxes = boxes[ix.data, :]

        # Keep track of which box is mapped to which level
        box_to_level.append(ix.data)

        # Stop gradient propogation to ROI proposals
        level_boxes = level_boxes.detach()

        # Crop and Resize
        # From Mask R-CNN paper: "We sample four regular locations, so
        # that we can evaluate either max or average pooling. In fact,
        # interpolating only a single value at each bin center (without
        # pooling) is nearly as effective."
        #
        # Here we use the simplified approach of a single value per bin,
        # which is how it's done in tf.crop_and_resize()
        # Result: [batch * num_boxes, pool_height, pool_width, channels]
        ind = Variable(torch.zeros(level_boxes.size()[0]),requires_grad=False).int()
        if level_boxes.is_cuda:
            ind = ind.cuda()
        feature_maps[i] = feature_maps[i].unsqueeze(0)  #CropAndResizeFunction needs batch dimension
        pooled_features = CropAndResizeFunction(pool_size, pool_size, 0)(feature_maps[i], level_boxes, ind)
        pooled.append(pooled_features)

    # Pack pooled features into one tensor
    pooled = torch.cat(pooled, dim=0)

    # Pack box_to_level mapping into one array and add another
    # column representing the order of pooled boxes
    box_to_level = torch.cat(box_to_level, dim=0)

    # Rearrange pooled features to match the order of the original boxes
    _, box_to_level = torch.sort(box_to_level)
    pooled = pooled[box_to_level, :, :]

    return pooled

def bbox_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    """
    # 1. Tile boxes2 and repeate boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeate() so simulate it
    # using tf.tile() and tf.reshape.
    boxes1_repeat = boxes2.size()[0]
    boxes2_repeat = boxes1.size()[0]
    boxes1 = boxes1.repeat(1,boxes1_repeat).view(-1,4)
    boxes2 = boxes2.repeat(boxes2_repeat,1)

    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = boxes1.chunk(4, dim=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = boxes2.chunk(4, dim=1)
    y1 = torch.max(b1_y1, b2_y1)[:, 0]
    x1 = torch.max(b1_x1, b2_x1)[:, 0]
    y2 = torch.min(b1_y2, b2_y2)[:, 0]
    x2 = torch.min(b1_x2, b2_x2)[:, 0]
    zeros = Variable(torch.zeros(y1.size()[0]), requires_grad=False)
    if y1.is_cuda:
        zeros = zeros.cuda()
    intersection = torch.max(x2 - x1, zeros) * torch.max(y2 - y1, zeros)

    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area[:,0] + b2_area[:,0] - intersection

    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = iou.view(boxes2_repeat, boxes1_repeat)

    return overlaps
