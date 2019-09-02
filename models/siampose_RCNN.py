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
from utils.image import draw_boxes
from utils.pose_evaluate import accuracy
from models.losses import FocalLoss, RegL1Loss, RegLoss, RegWeightedL1Loss
from models.utils import _sigmoid, proposal_layer, roi_align, generate_target_gt


class PoseLoss(torch.nn.Module):
    def __init__(self, opt):
        super(PoseLoss, self).__init__()
        self.crit_hm_hp = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_kp = RegWeightedL1Loss() if not opt.dense_hp else \
            torch.nn.L1Loss(reduction='sum')
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.opt = opt

    def forward(self, outputs, batch, ind):
        global pose_iter
        opt = self.opt
        hp_loss, hm_hp_loss, hp_offset_loss = 0, 0, 0
        output = outputs[0]
        ind = ind.long()
        # batch['hps_mask'] = batch['hps_mask'].index_select(0, ind)
        batch['ind'] = batch['ind'].index_select(0, ind)
        # batch['hps'] = batch['hps'].index_select(0, ind)
        batch['hp_mask'] = batch['hp_mask'].index_select(0, ind)
        batch['hp_ind'] = batch['hp_ind'].index_select(0, ind)
        batch['hp_offset'] = batch['hp_offset'].index_select(0, ind)
        # print('hps_mask shape: {}, ind shape: {} , hps shape: {}'.format(batch['hps_mask'].shape,
        #        batch['ind'].shape, batch['hps'].shape))
        # print('output hps shape: ', output['hps'].shape)
        batch['hm_hp'] = batch['hm_hp'].index_select(0, ind)
        if opt.hm_hp and not opt.mse_loss:
            output['hm_hp'] = _sigmoid(output['hm_hp'])

        # if opt.dense_hp:
        #     mask_weight = batch['dense_hps_mask'].sum() + 1e-4
        #     hp_loss += (self.crit_kp(output['hps'] * batch['dense_hps_mask'],
        #                 batch['dense_hps'] * batch['dense_hps_mask']) /
        #                 mask_weight)
        # else:
        #     hp_loss += self.crit_kp(output['hps'], batch['hps_mask'],
        #                             batch['ind'], batch['hps'])

        if opt.reg_hp_offset and opt.off_weight > 0:
            hp_offset_loss += self.crit_reg(
              output['hp_offset'], batch['hp_mask'],
              batch['hp_ind'], batch['hp_offset'])
        # print('out hm hp shape: {}, batch hm hp shape: {}'.format(output['hm_hp'].shape, batch['hm_hp']))
        if opt.hm_hp and opt.hm_hp_weight > 0:
            hm_hp_loss += self.crit_hm_hp(
              output['hm_hp'], batch['hm_hp'])

        loss = opt.hm_hp_weight * hm_hp_loss + opt.off_weight * hp_offset_loss
        # opt.hp_weight * hp_loss + \

        loss_stats = {'loss': loss,  # 'hp_loss': hp_loss,
                      'hm_hp_loss': hm_hp_loss, 'hp_offset_loss': hp_offset_loss}

        return loss, loss_stats

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
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints

class SiamMask(nn.Module):
    def __init__(self, opts=None, anchors=None, o_sz=63, g_sz=127):
        super(SiamMask, self).__init__()
        self.opt = opts
        self.anchor = anchors
        self.anchor_num = anchors.anchor_num
        self.features = None
        self.rpn_model = None
        self.mask_model = None
        self.o_sz = o_sz
        self.g_sz = g_sz
        self.output_size = opts.output_size
        self.upSample = nn.UpsamplingBilinear2d(size=[g_sz, g_sz])
        self.kp_criterion = PoseLoss(self.opt)
        self.bs = self.opt.batch
        self.anchors_preprocess()

        self.debug = opts.debug

    def anchors_preprocess(self):
        """
        Returns:
            boxes: [batch//gpu_count, *height*width*anchors, 4]
        """
        all_anchors = self.anchor.all_anchors[0]
        gpu_count = torch.cuda.device_count()
        self.all_anchors = torch.from_numpy(all_anchors).float().cuda().detach()
        # x1,y1,x2,y2
        # anchors: [4, anchors, height, width, (x1, y1, x2, y2)]
        assert self.bs // gpu_count > 0
        boxes = self.all_anchors.expand(self.bs//gpu_count, -1, -1, -1, -1)
        boxes = boxes.permute(0, 3, 4, 2, 1).contiguous().view(self.bs//gpu_count, -1, 4)
        # self.all_anchors = torch.from_numpy(all_anchors).float().cuda()
        # self.all_anchors = [self.all_anchors[i] for i in range(4)]
        self.anchors = boxes
        # return boxes

    def proposal_preprocess(self, rpn_pred_score, rpn_pred_loc):
        """
        Inputs:
            rpn_pred_score: [batch, 2*anchors, height, width, (fg prob, bg prob)]
            rpn_pred_loc: [batch, 4*anchors, height, width]
            # gt_kps: [batch, num_keypoints, height, width]

        Returns:
            scores: [batch, height*width*anchors]
            deltas: [batch, height*width*anchors, 4]
            # kps:    [batch, height*width*anchors, num_keypoints]
            # masks:  to be done
        """
        bs = rpn_pred_score.size(0)
        scores = rpn_pred_score[:, :, :, :, 1]
        scores = scores.transpose(1, 3).contiguous().view(bs, -1)

        deltas = rpn_pred_loc
        deltas = deltas.view(-1, 4, self.anchor_num, deltas.size(-2), deltas.size(-1))
        deltas = deltas.permute(0, 3, 4, 2, 1).contiguous()
        deltas = deltas.view(bs, -1, deltas.size(4))

        # num_kps = gt_kps.size(1)
        # kps = gt_kps.transpose(1, 3).contiguous()
        # kps = kps.expand(-1, -1, -1, self.anchor_num, num_kps)
        # kps = kps.view(-1, num_kps)

        return [scores, deltas]

    def feature_extractor(self, x):
        return self.features(x)

    def rpn(self, template, search):
        pred_cls, pred_loc = self.rpn_model(template, search)
        return pred_cls, pred_loc

    def mask(self, template, search):
        pred_mask = self.mask_model(template, search)
        return pred_mask

    def _add_rpn_loss(self, label_cls, label_loc, lable_loc_weight,
                      rpn_pred_cls, rpn_pred_loc):
        rpn_loss_cls = select_cross_entropy_loss(rpn_pred_cls, label_cls)
        # print('label_cls shape: ', label_cls.shape)
        # print('label_cls positive: ', torch.nonzero(label_cls > 0.5))
        rpn_loss_loc = weight_l1_loss(rpn_pred_loc, label_loc, lable_loc_weight)

        return rpn_loss_cls, rpn_loss_loc

    def run(self, template, search, softmax=False):
        """
        run network
        """
        template_feature, template_deeper_feature = self.feature_extractor(template)
        search_feature, p4_feat = self.features.forward_all(search)

        rpn_pred_cls, rpn_pred_loc = self.rpn(template_feature, search_feature)
        rpn_pred_cls_sfmax = None
        if softmax:
            rpn_pred_cls_sfmax = self.softmax(rpn_pred_cls, log=True)
        rpn_pred_cls = self.softmax(rpn_pred_cls, log=False)

        return rpn_pred_cls_sfmax, rpn_pred_loc, template_deeper_feature, search_feature, rpn_pred_cls, p4_feat

    def softmax(self, cls, log=True):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2 // 2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        if log is True:
            cls = F.log_softmax(cls, dim=4)
        else:
            cls = F.softmax(cls, dim=4)
        return cls

    def forward(self, rpn_input, kp_input):
        """
        :param rpn_input: dict of input with keys of:
                'template': [b, 3, h1, w1], input template image.
                'search': [b, 3, h2, w2], input search image.
                'label_cls':[b, max_num_gts, 5] or None(self.training==False),
                                     each gt contains x1,y1,x2,y2,class.
        :return: dict of loss, predict, accuracy
        """
        template = rpn_input['template']
        search = rpn_input['search']
        anchors = self.all_anchors

        label_cls = rpn_input['label_cls']
        label_loc = rpn_input['label_loc']
        # label_mask = rpn_input['label_mask']
        lable_loc_weight = rpn_input['label_loc_weight']
        # anchors = rpn_input['anchors']

        rpn_pred_cls, rpn_pred_loc, template_feature, search_feature, rpn_pred_score, p4_feat = \
            self.run(template, search, softmax=True)

        proposals = self.proposal_preprocess(rpn_pred_score, rpn_pred_loc)

        normalized_boxes, boxes_ind, box_flag = proposal_layer(proposals, self.anchors, args=self.opt)
        template_feature = torch.index_select(template_feature, 0, boxes_ind.long())
        # print('per batch nms boxes shape: ', normalized_boxes.shape)
        # normalized_boxes = normalized_boxes.view(-1, normalized_boxes.size(-1))
        # print('normalized bbox: ', normalized_boxes)
        if box_flag:
            pooled_features = roi_align([normalized_boxes, p4_feat, boxes_ind], 7)
            # print('poolded features shape: ', pooled_features.shape)
            pred_kp = self.kp_model(pooled_features, template_feature)
            # gt_sample = kp_input['hm_hp']
            gt_hm_hp = search   # kp_input['hm_hp']
            # gt_hm_hp = generate_target_gt(gt_sample, normalized_boxes, boxes_ind, self.output_size)
            # kp_input['hm_hp'] = gt_hm_hp
        else:
            print('no box flag')
            # 'hps': 34, 'hm_hp': 17, 'hp_offset': 2
            pred_hm_hp = torch.zeros(p4_feat.size(0), 17, 56, 56).cuda()
            pred_hp_offset = torch.zeros(p4_feat.size(0), 2, 56, 56).cuda()
            pred_kp = [{'hm_hp': pred_hm_hp, 'hp_offset': pred_hp_offset}]
        outputs = dict()

        outputs['predict'] = [rpn_pred_cls, rpn_pred_loc, pred_kp,
                              template_feature, search_feature, rpn_pred_score, normalized_boxes]

        rpn_loss_cls, rpn_loss_loc = \
            self._add_rpn_loss(label_cls, label_loc, lable_loc_weight,
                               rpn_pred_cls, rpn_pred_loc)
        pred_hm = pred_kp[0]['hm_hp'].detach().cpu().numpy()
        gt_hm = kp_input['hm_hp'].detach().cpu().numpy()
        pck = accuracy(pred_hm, gt_hm)
        acc = torch.Tensor(pck[0]).float().cuda()
        avg_acc = torch.Tensor([pck[1]]).float().cuda()
        outputs['accuracy'] = [acc, avg_acc]

        if box_flag:
            kp_loss, kp_loss_status = self.kp_criterion(pred_kp, kp_input, boxes_ind)
        else:
            kp_loss = torch.zeros(1, 1).cuda()
            kp_loss_status = {'loss': torch.zeros(1, 1).cuda(), 'hp_loss': torch.zeros(1, 1).cuda(),
                              'hm_hp_loss': torch.zeros(1, 1).cuda(), 'hp_offset_loss': torch.zeros(1, 1).cuda()}
        outputs['losses'] = [rpn_loss_cls, rpn_loss_loc, kp_loss, kp_loss_status]

        if self.debug:
            # draw roi boxes on the search images
            box_imgs = draw_boxes(search, normalized_boxes, boxes_ind)
            if box_flag:
                roi_imgs = roi_align([normalized_boxes, search, boxes_ind], 56)
                roi_imgs = roi_imgs.transpose(1, 3)
                hp_imgs = gt_hm_hp
            else:
                roi_imgs = None
            outputs['debug'] = [torch.from_numpy(box_imgs).cuda(), roi_imgs, hp_imgs]

        # outputs['accuracy'] = [iou_acc_mean, iou_acc_5, iou_acc_7]
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
    # print('pred: ', pred.shape, pred)
    # print('label: ', label.shape, label)

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
