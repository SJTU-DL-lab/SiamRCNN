# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.anchors import Anchors
import matplotlib.pyplot as plt

class JointsSL1Loss(nn.Module):

    def __init__(self, use_target_weight):
        super(JointsSL1Loss, self).__init__()
        self.criterion = nn.SmoothL1Loss()
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        target = target.float()
        # output = output.float()
        # target_weight = target_weight.float()
        target_weight = torch.unsqueeze(target_weight, -1)
        batch_size = output.size(0)
        num_joints = 17
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()

            if self.use_target_weight:
                loss += self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class JointsMSELoss(nn.Module):

    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss()
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        target_weight = torch.unsqueeze(target_weight, -1)
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()

            if self.use_target_weight:
                loss += self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class SiamMask(nn.Module):
    def __init__(self, anchors=None, o_sz=63, g_sz=127):
        super(SiamMask, self).__init__()
        self.anchors = anchors  # anchor_cfg
        self.anchor_num = len(self.anchors["ratios"]) * len(self.anchors["scales"])
        self.anchor = Anchors(anchors)
        self.features = None
        self.rpn_model = None
        self.mask_model = None
        self.o_sz = o_sz
        self.g_sz = g_sz
        self.upSample = nn.UpsamplingBilinear2d(size=[g_sz, g_sz])
        self.kp_criterion = JointsSL1Loss(True)
        self.heatmap_criterion = JointsMSELoss(False)

        self.all_anchors = None

    def set_all_anchors(self, image_center, size):
        # cx,cy,w,h
        if not self.anchor.generate_all_anchors(image_center, size):
            return
        all_anchors = self.anchor.all_anchors[1]  # cx, cy, w, h
        self.all_anchors = torch.from_numpy(all_anchors).float().cuda()
        self.all_anchors = [self.all_anchors[i] for i in range(4)]

    def feature_extractor(self, x):
        return self.features(x)

    def rpn(self, template, search):
        pred_cls, pred_loc = self.rpn_model(template, search)
        return pred_cls, pred_loc

    def mask(self, template, search):
        kp_coord, kp_heatmap = self.mask_model(template, search)
        return kp_coord, kp_heatmap

    def _add_rpn_loss(self, label_cls, label_loc, lable_loc_weight, label_kp, label_heatmap, label_mask_weight,
                      rpn_pred_cls, rpn_pred_loc, kp_coord, kp_heatmap, kp_weight, kp_criterion, heatmap_criterion):
        rpn_loss_cls = select_cross_entropy_loss(rpn_pred_cls, label_cls)

        rpn_loss_loc = weight_l1_loss(rpn_pred_loc, label_loc, lable_loc_weight)

        rpn_loss_kp = select_kp_logistic_loss(kp_coord,
                                                label_kp,
                                                label_mask_weight,
                                                kp_weight,
                                                kp_criterion)

        rpn_loss_heatmap, kp_pred, htmap_pred = select_heatmap_logistic_loss(kp_heatmap,
                                                     label_heatmap,
                                                     label_mask_weight,
                                                     kp_weight,
                                                     heatmap_criterion)

        return rpn_loss_cls, rpn_loss_loc, rpn_loss_kp, rpn_loss_heatmap, kp_pred, htmap_pred

    def run(self, template, search, softmax=False):
        """
        run network
        """
        template_feature = self.feature_extractor(template)
        search_feature = self.feature_extractor(search)
        rpn_pred_cls, rpn_pred_loc = self.rpn(template_feature, search_feature)
        kp_coord, kp_heatmap = self.mask(template_feature, search_feature)

        if softmax:
            rpn_pred_cls = self.softmax(rpn_pred_cls)
        return rpn_pred_cls, rpn_pred_loc, kp_coord, kp_heatmap, template_feature, search_feature

    def softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2 // 2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, input):
        """
        :param input: dict of input with keys of:
                'template': [b, 3, h1, w1], input template image.
                'search': [b, 3, h2, w2], input search image.
                'label_cls':[b, max_num_gts, 5] or None(self.training==False),
                                     each gt contains x1,y1,x2,y2,class.
        :return: dict of loss, predict, accuracy
        """
        template = input['template']
        search = input['search']
        if self.training:
            label_cls = input['label_cls']
            label_loc = input['label_loc']
            lable_loc_weight = input['label_loc_weight']
            label_mask = input['label_mask']
            label_mask_weight = input['label_mask_weight']
            label_kp_weight = input['label_kp_weight']
            label_kp = input['label_kp']

        rpn_pred_cls, rpn_pred_loc, kp_coord, kp_heatmap, template_feature, search_feature = \
            self.run(template, search, softmax=self.training)

        outputs = dict()

        outputs['predict'] = [rpn_pred_loc, rpn_pred_cls, kp_coord, kp_heatmap, template_feature, search_feature]

        if self.training:
            rpn_loss_cls, rpn_loss_loc, rpn_loss_kp, rpn_loss_heatmap, kp_pred, htmap_pred = \
                self._add_rpn_loss(label_cls, label_loc, lable_loc_weight, label_kp, label_mask, label_mask_weight,
                                   rpn_pred_cls, rpn_pred_loc, kp_coord, kp_heatmap,
                                   label_kp_weight, self.kp_criterion, self.heatmap_criterion)
            outputs['losses'] = [rpn_loss_cls, rpn_loss_loc, rpn_loss_kp, rpn_loss_heatmap]

        outputs['predict'].extend([kp_pred, htmap_pred])

        return outputs

    def template(self, z):
        self.zf = self.feature_extractor(z)
        cls_kernel, loc_kernel = self.rpn_model.template(self.zf)
        return cls_kernel, loc_kernel

    def track(self, x, cls_kernel=None, loc_kernel=None, softmax=False):
        xf = self.feature_extractor(x)
        rpn_pred_cls, rpn_pred_loc = self.rpn_model.track(xf, cls_kernel, loc_kernel)
        if softmax:
            rpn_pred_cls = self.softmax(rpn_pred_cls)
        return rpn_pred_cls, rpn_pred_loc


def get_cls_loss(pred, label, select):
    if select.nelement() == 0: return pred.sum() * 0.
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)

    return F.nll_loss(pred, label)


def select_cross_entropy_loss(pred, label):
    pred = pred.view(-1, 2)
    label = label.view(-1)
    pos = Variable(label.data.eq(1).nonzero().squeeze()).cuda()
    neg = Variable(label.data.eq(0).nonzero().squeeze()).cuda()

    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5


def weight_l1_loss(pred_loc, label_loc, loss_weight):
    """
    :param pred_loc: [b, 4k, h, w]
    :param label_loc: [b, 4k, h, w]
    :param loss_weight:  [b, k, h, w]
    :return: loc loss value
    """
    b, _, sh, sw = pred_loc.size()
    pred_loc = pred_loc.view(b, 4, -1, sh, sw)
    diff = (pred_loc - label_loc).abs()
    diff = diff.sum(dim=1).view(b, -1, sh, sw)
    loss = diff * loss_weight
    return loss.sum().div(b)


def select_kp_logistic_loss(p_m, mask, weight, kp_weight, criterion, o_sz=63, g_sz=127):
    # mask = mask[:, 0, :, :]
    # mask = mask.unsqueeze(1)
    # print('mask shape: ', mask.shape)
    # print('pred mask shape: ', p_m.shape)
    # print('mask weight shape: ', weight.shape)
    # print('kp weight shape: ', kp_weight.shape)

    kp_weight_pos = kp_weight.view(kp_weight.size(0), 1, 1, 1, -1)
    kp_weight_pos = kp_weight_pos.expand(-1,
                                         weight.size(1),
                                         weight.size(2),
                                         weight.size(3),
                                         -1).contiguous()
    # (bs, 1, 25, 25, 17)
    kp_weight_pos = kp_weight_pos.view(-1, 17)

    mask_weight_pos = mask.view(mask.size(0), 1, 1, 1, -1, 2)
    mask_weight_pos = mask_weight_pos.expand(-1,
                                         weight.size(1),
                                         weight.size(2),
                                         weight.size(3),
                                         -1, -1).contiguous()
    # (bs, 1, 25, 25, 17)
    mask_weight_pos = mask_weight_pos.view(-1, 17, 2)

    weight = weight.view(-1)
    pos = Variable(weight.data.eq(1).nonzero().squeeze())
    kp_weight = torch.index_select(kp_weight_pos, 0, pos)
    mask = torch.index_select(mask_weight_pos, 0, pos)
    # print('pose shape: ', pos.shape)
    if pos.nelement() == 0: return p_m.sum() * 0

    if len(p_m.shape) == 4:
        p_m = p_m.permute(0, 2, 3, 1).contiguous().view(-1, 17, 2)
        # print('atf pred mask shape: ', p_m.shape)
        p_m = torch.index_select(p_m, 0, pos)
        # print('atf selected pred mask shape: ', p_m.shape)
    else:
        p_m = torch.index_select(p_m, 0, pos)

    loss = criterion(p_m, mask, kp_weight)

    return loss

def select_heatmap_logistic_loss(p_m, mask, weight, kp_weight, criterion, o_sz=63, g_sz=127):
    kp_weight_pos = kp_weight.view(kp_weight.size(0), 1, 1, 1, -1)
    kp_weight_pos = kp_weight_pos.expand(-1,
                                         weight.size(1),
                                         weight.size(2),
                                         weight.size(3),
                                         -1).contiguous()
    # (bs, 1, 25, 25, 17)
    kp_weight_pos = kp_weight_pos.view(-1, 1)
    weight = weight.view(-1)
    pos = Variable(weight.data.eq(1).nonzero().squeeze())
    # print('pose shape: ', pos.shape)
    if pos.nelement() == 0: return p_m.sum() * 0

    if len(p_m.shape) == 4:
        p_m = p_m.permute(0, 2, 3, 1).contiguous().view(-1, 1, o_sz, o_sz)
        # print('atf pred mask shape: ', p_m.shape)
        p_m = torch.index_select(p_m, 0, pos)
        # print('atf selected pred mask shape: ', p_m.shape)
        p_m = nn.UpsamplingBilinear2d(size=[g_sz, g_sz])(p_m)
        # p_m = p_m.view(-1, g_sz * g_sz * 17)
    else:
        p_m = torch.index_select(p_m, 0, pos)
        p_m = p_m.view(-1, 1, g_sz, g_sz)

    kp_weight = torch.index_select(kp_weight_pos, 0, pos)

    mask_uf = F.unfold(mask, (g_sz, g_sz), padding=32, stride=8)
    # print('mask uf shape: ', mask_uf.shape)
    mask_uf = torch.transpose(mask_uf, 1, 2).contiguous().view(-1, g_sz * g_sz * 1)
    # print('transpose mask uf shape: ', mask_uf.shape)

    mask_uf = torch.index_select(mask_uf, 0, pos)
    mask_uf = mask_uf.view(-1, 1, g_sz, g_sz)
    # loss = F.soft_margin_loss(p_m, mask_uf)
    loss = criterion(p_m, mask_uf, kp_weight)

    return loss, p_m, mask_uf

def iou_measure(pred, label):
    pred = pred.ge(0)
    mask_sum = pred.eq(1).add(label.eq(1))
    intxn = torch.sum(mask_sum == 2, dim=1).float()
    union = torch.sum(mask_sum > 0, dim=1).float()
    iou = intxn / union
    return torch.mean(iou), (torch.sum(iou > 0.5).float() / iou.shape[0]), (torch.sum(iou > 0.7).float() / iou.shape[0])


if __name__ == "__main__":
    p_m = torch.randn(4, 63 * 63, 25, 25)
    cls = torch.randn(4, 1, 25, 25) > 0.9
    mask = torch.randn(4, 1, 255, 255) * 2 - 1

    loss = select_mask_logistic_loss(p_m, mask, cls)
    print(loss)
