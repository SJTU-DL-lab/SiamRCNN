from models.siampose_RCNN_base import SiamMask
from models.features import MultiStageFeature
from models.rpn import RPN, DepthCorr
from models.mask import Mask
# from models.DCNv2.dcn_v2 import DCN
# from models.DCN.modules.modulated_dcn import ModulatedDeformConvPack
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.load_helper import load_pretrain
from resnet import resnet50

# import sys
# from gpu_profile import gpu_profile

BN_MOMENTUM = 0.1
# DCN = ModulatedDeformConvPack

# sys.settrace(gpu_profile)
class ResDownS(nn.Module):
    def __init__(self, inplane, outplane):
        super(ResDownS, self).__init__()
        self.downsample = nn.Sequential(
                nn.Conv2d(inplane, outplane, kernel_size=1, bias=False),
                nn.BatchNorm2d(outplane))

    def forward(self, x):
        x = self.downsample(x)
        if x.size(3) < 20:
            l = 4
            r = -4
            x = x[:, :, l:r, l:r]
        return x


class ResDown(MultiStageFeature):
    def __init__(self, pretrain=False):
        super(ResDown, self).__init__()
        self.features = resnet50(layer3=True, layer4=False)
        if pretrain:
            load_pretrain(self.features, '../resnet.model')

        self.downsample = ResDownS(1024, 256)
        # self.downsample_p4 = ResDownS(2048, 1024)
        self.hidden_layer = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                )
        self.big_kernel = nn.Conv2d(512, 512, kernel_size=7, bias=False, groups=512)
        self.deeper_layer = nn.Sequential(
                                   nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(inplace=True)
                            )

        self.layers = [self.downsample, # self.downsample_p4,
                       self.features.layer2, self.features.layer3,
                       self.hidden_layer, self.big_kernel, self.deeper_layer] #, self.features.layer4]
        self.train_nums = [6, 6]
        self.change_point = [0, 0.5]

        self.unfix(0.0)

    def param_groups(self, start_lr, feature_mult=1):
        lr = start_lr * feature_mult

        def _params(module, mult=1):
            params = list(filter(lambda x:x.requires_grad, module.parameters()))
            if len(params):
                return [{'params': params, 'lr': lr * mult}]
            else:
                return []

        groups = []
        groups += _params(self.downsample)
        groups += _params(self.features, 0.1)
        return groups

    def forward(self, x):
        output = self.features(x)
        p3 = self.downsample(output[-2])

        p3_deeper = self.deeper_layer(p3)
        return p3, p3_deeper

    def forward_all(self, x):
        output = self.features(x)
        p3 = self.downsample(output[-2])
        # p4 = self.downsample_p4(output[-1])
        kp_feat = self.hidden_layer(p3)
        kp_feat = self.big_kernel(kp_feat)
        return p3, kp_feat


class UP(RPN):
    def __init__(self, anchor_num=5, feature_in=256, feature_out=256):
        super(UP, self).__init__()

        self.anchor_num = anchor_num
        self.feature_in = feature_in
        self.feature_out = feature_out

        self.cls_output = 2 * self.anchor_num
        self.loc_output = 4 * self.anchor_num

        self.cls = DepthCorr(feature_in, feature_out, self.cls_output)
        self.loc = DepthCorr(feature_in, feature_out, self.loc_output)

    def forward(self, z_f, x_f):
        cls = self.cls(z_f, x_f)
        loc = self.loc(z_f, x_f)
        return cls, loc


class Custom(SiamMask):
    def __init__(self, pretrain=False, **kwargs):
        super(Custom, self).__init__(**kwargs)
        # self.opt = opt
        self.features = ResDown(pretrain=pretrain)
        self.rpn_model = UP(anchor_num=self.anchor_num, feature_in=256, feature_out=256)
        # self.kp_corr = KpCorr()
        # self.kp_model = Center_pose_head()

    def template(self, template):
        self.zf = self.features(template)

    def track(self, search):
        search = self.features(search)
        rpn_pred_cls, rpn_pred_loc = self.rpn(self.zf, search)
        return rpn_pred_cls, rpn_pred_loc

    def track_mask(self, search):
        search = self.features(search)
        rpn_pred_cls, rpn_pred_loc = self.rpn(self.zf, search)
        pred_mask = self.mask(self.zf, search)
        return rpn_pred_cls, rpn_pred_loc, pred_mask


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def fill_ones_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.ones(m.weight)

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]