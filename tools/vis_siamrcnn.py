# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import argparse
import logging
import os
import cv2
import shutil
import time
import json
import math
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import copy

from torch.utils.data import DataLoader
from torch.autograd import Variable

from utils.log_helper import init_log, print_speed, add_file_handler, Dummy
from utils.load_helper import load_pretrain, restore_from
from utils.average_meter_helper import AverageMeter
from utils.image import save_gt_pred_heatmaps

from datasets.siam_rcnn_dataset import DataSets
from utils.lr_helper import build_lr_scheduler
from tensorboardX import SummaryWriter

from utils.config_helper import load_config
from torch.utils.collect_env import get_pretty_env_info
from torchvision.transforms import ToTensor

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch Tracking SiamMask Training')

parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--clip', default=10.0, type=float,
                    help='gradient clip value')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', default='',
                    help='use pre-trained model')
parser.add_argument('--config', dest='config', required=True,
                    help='hyperparameter of SiamMask in json format')
parser.add_argument('--arch', dest='arch', default='', choices=['Custom',],
                    help='architecture of pretrained model')
parser.add_argument('-l', '--log', default="log.txt", type=str,
                    help='log file')
parser.add_argument('-s', '--save_dir', default='snapshot', type=str,
                    help='save dir')
parser.add_argument('--log-dir', default='board', help='TensorBoard log dir')
# multi_pose
parser.add_argument('--dense_hp', action='store_true',
                         help='apply weighted pose regression near center '
                              'or just apply regression on center point.')
parser.add_argument('--not_hm_hp', action='store_true',
                         help='not estimate human joint heatmap, '
                              'directly use the joint offset from center.')
parser.add_argument('--not_reg_hp_offset', action='store_true',
                         help='not regress local offset for '
                              'human joint heatmaps.')
parser.add_argument('--not_reg_bbox', action='store_true',
                         help='not regression bounding box size.')
# losses
parser.add_argument('--mse_loss', action='store_true',
                             help='use mse loss or focal loss to train '
                                  'keypoint heatmaps.')
parser.add_argument('--reg_loss', default='l1',
                             help='regression loss: sl1 | l1 | l2')
# parser.add_argument('--hm_weight', type=float, default=1,
#                              help='loss weight for keypoint heatmaps.')
parser.add_argument('--off_weight', type=float, default=1,
                         help='loss weight for keypoint local offsets.')
# parser.add_argument('--wh_weight', type=float, default=0.1,
#                          help='loss weight for bounding box size.')
parser.add_argument('--hp_weight', type=float, default=1,
                             help='loss weight for human pose offset.')
parser.add_argument('--hm_hp_weight', type=float, default=1,
                             help='loss weight for human keypoint heatmap.')

best_acc = 0.

def collect_env_info():
    env_str = get_pretty_env_info()
    env_str += "\n        OpenCV ({})".format(cv2.__version__)
    return env_str


def build_data_loader(cfg):
    logger = logging.getLogger('global')

    logger.info("build train dataset")  # train_dataset
    train_set = DataSets(cfg['train_datasets'], cfg['anchors'], args.epochs)
    train_set.shuffle()

    logger.info("build val dataset")  # val_dataset
    if not 'val_datasets' in cfg.keys():
        cfg['val_datasets'] = cfg['train_datasets']
    val_set = DataSets(cfg['val_datasets'], cfg['anchors'])
    val_set.shuffle()

    train_loader = DataLoader(train_set, batch_size=args.batch, num_workers=args.workers,
                              pin_memory=True, sampler=None)
    val_loader = DataLoader(val_set, batch_size=args.batch, num_workers=args.workers,
                            pin_memory=True, sampler=None)

    logger.info('build dataset done')
    return train_loader, val_loader


def build_opt_lr(model, cfg, args, epoch):
    backbone_feature = model.features.param_groups(cfg['lr']['start_lr'], cfg['lr']['feature_lr_mult'])
    if len(backbone_feature) == 0:
        trainable_params = model.rpn_model.param_groups(cfg['lr']['start_lr'], cfg['lr']['rpn_lr_mult'], 'mask')
    else:
        trainable_params = backbone_feature + \
                           model.rpn_model.param_groups(cfg['lr']['start_lr'], cfg['lr']['rpn_lr_mult']) + \
                           model.kp_model.param_groups(cfg['lr']['start_lr'], cfg['lr']['mask_lr_mult'])

    optimizer = torch.optim.SGD(trainable_params, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = build_lr_scheduler(optimizer, cfg['lr'], epochs=args.epochs)

    lr_scheduler.step(epoch)

    return optimizer, lr_scheduler


def main():
    global args, best_acc, tb_writer, logger
    args = parser.parse_args()
    args = args_process(args)

    init_log('global', logging.INFO)

    if args.log != "":
        add_file_handler('global', args.log, logging.INFO)

    logger = logging.getLogger('global')
    logger.info("\n" + collect_env_info())
    logger.info(args)

    cfg = load_config(args)
    logger.info("config \n{}".format(json.dumps(cfg, indent=4)))

    if args.log_dir:
        tb_writer = SummaryWriter(args.log_dir)
    else:
        tb_writer = Dummy()

    # build dataset
    train_loader, val_loader = build_data_loader(cfg)

    args.img_size = int(cfg['train_datasets']['search_size'])
    args.nms_threshold = float(cfg['train_datasets']['RPN_NMS'])
    if args.arch == 'Custom':
        from custom import Custom
        model = Custom(pretrain=True, opts=args, anchors=cfg['anchors'])
    else:
        exit()
    logger.info(model)

    if args.pretrained:
        model = load_pretrain(model, args.pretrained)

    model = model.cuda()
    dist_model = torch.nn.DataParallel(model, list(range(torch.cuda.device_count()))).cuda()

    if args.resume and args.start_epoch != 0:
        model.features.unfix((args.start_epoch - 1) / args.epochs)

    optimizer, lr_scheduler = build_opt_lr(model, cfg, args, args.start_epoch)
    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume), '{} is not a valid file'.format(args.resume)
        model, optimizer, args.start_epoch, best_acc, arch = restore_from(model, optimizer, args.resume)
        dist_model = torch.nn.DataParallel(model, list(range(torch.cuda.device_count()))).cuda()

    logger.info(lr_scheduler)

    logger.info('model prepare done')

    train(train_loader, dist_model, optimizer, lr_scheduler, args.start_epoch, cfg)


def BNtoFixed(m):
    class_name = m.__class__.__name__
    if class_name.find('BatchNorm') != -1:
        m.eval()


def train(train_loader, model, optimizer, lr_scheduler, epoch, cfg):
    global tb_index, best_acc, cur_lr, logger
    cur_lr = lr_scheduler.get_cur_lr()
    logger = logging.getLogger('global')
    avg = AverageMeter()
    model.train()
    # model.module.features.eval()
    # model.module.rpn_model.eval()
    # model.module.features.apply(BNtoFixed)
    # model.module.rpn_model.apply(BNtoFixed)
    #
    # model.module.mask_model.train()
    # model.module.refine_model.train()
    model = model.cuda()
    end = time.time()

    def is_valid_number(x):
        return not(math.isnan(x) or math.isinf(x) or x > 1e4)

    num_per_epoch = len(train_loader.dataset) // args.epochs // args.batch
    start_epoch = epoch
    epoch = epoch
    with torch.no_grad():
        for iter, input in enumerate(train_loader):
            if iter > 100:
                break

            if epoch != iter // num_per_epoch + start_epoch:  # next epoch
                epoch = iter // num_per_epoch + start_epoch

                if epoch == args.epochs:
                    return

                if model.module.features.unfix(epoch/args.epochs):
                    logger.info('unfix part model.')
                    optimizer, lr_scheduler = build_opt_lr(model.module, cfg, args, epoch)

                lr_scheduler.step(epoch)
                cur_lr = lr_scheduler.get_cur_lr()

                logger.info('epoch:{}'.format(epoch))

            tb_index = iter
            if iter % num_per_epoch == 0 and iter != 0:
                for idx, pg in enumerate(optimizer.param_groups):
                    logger.info("epoch {} lr {}".format(epoch, pg['lr']))
                    tb_writer.add_scalar('lr/group%d' % (idx+1), pg['lr'], tb_index)

            data_time = time.time() - end
            avg.update(data_time=data_time)
            x_rpn = {
                'cfg': cfg,
                'template': torch.autograd.Variable(input[0]).cuda(),
                'search': torch.autograd.Variable(input[1]).cuda(),
                'label_cls': torch.autograd.Variable(input[2]).cuda(),
                'label_loc': torch.autograd.Variable(input[3]).cuda(),
                'label_loc_weight': torch.autograd.Variable(input[4]).cuda(),
                'label_mask': torch.autograd.Variable(input[6]).cuda()
            }
            x_kp = input[7]
            x_kp = {x: torch.autograd.Variable(y).cuda() for x, y in x_kp.items()}
            x_rpn['anchors'] = train_loader.dataset.anchors.all_anchors[0]

            outputs = model(x_rpn, x_kp)
            roi_box = outputs['predict'][-1]
            pred_kp = outputs['predict'][2]['hm_hp']
            batch_img = x_rpn['search'].expand(x_kp['hm_hp'].size(0), -1, -1, -1)
            gt_img, pred_img = save_gt_pred_heatmaps(batch_img, x_kp['hm_hp'], pred_kp, 'test_imgs/test_{}.jpg'.format(iter))
            # rpn_pred_cls, rpn_pred_loc = outputs['predict'][:2]
            # rpn_pred_cls = outputs['predict'][-1]
            # anchors = train_loader.dataset.anchors.all_anchors[0]
            #
            # normalized_boxes = proposal_layer([rpn_pred_cls, rpn_pred_loc], anchors, config=cfg)
            # print('rpn_pred_cls: ', rpn_pred_cls.shape)

            rpn_cls_loss, rpn_loc_loss, kp_losses = torch.mean(outputs['losses'][0]),\
                                                        torch.mean(outputs['losses'][1]),\
                                                        outputs['losses'][3]
            kp_loss = torch.mean(kp_losses['loss'])
            kp_hp_loss = torch.mean(kp_losses['hp_loss'])
            kp_hm_hp_loss = torch.mean(kp_losses['hm_hp_loss'])
            kp_hp_offset_loss = torch.mean(kp_losses['hp_offset_loss'])

            # mask_iou_mean, mask_iou_at_5, mask_iou_at_7 = torch.mean(outputs['accuracy'][0]), torch.mean(outputs['accuracy'][1]), torch.mean(outputs['accuracy'][2])

            cls_weight, reg_weight, kp_weight = cfg['loss']['weight']

            loss = rpn_cls_loss * cls_weight + rpn_loc_loss * reg_weight + kp_loss * kp_weight

            optimizer.zero_grad()
            loss.backward()

            if cfg['clip']['split']:
                torch.nn.utils.clip_grad_norm_(model.module.features.parameters(), cfg['clip']['feature'])
                torch.nn.utils.clip_grad_norm_(model.module.rpn_model.parameters(), cfg['clip']['rpn'])
                torch.nn.utils.clip_grad_norm_(model.module.mask_model.parameters(), cfg['clip']['mask'])
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)  # gradient clip

            if is_valid_number(loss.item()):
                optimizer.step()

            siammask_loss = loss.item()

            batch_time = time.time() - end

            avg.update(batch_time=batch_time, rpn_cls_loss=rpn_cls_loss, rpn_loc_loss=rpn_loc_loss,
                       kp_hp_loss=kp_hp_loss, kp_hm_hp_loss=kp_hm_hp_loss, kp_hp_offset_loss=kp_hp_offset_loss,
                       kp_loss=kp_loss, siammask_loss=siammask_loss)
                       # mask_iou_mean=mask_iou_mean, mask_iou_at_5=mask_iou_at_5, mask_iou_at_7=mask_iou_at_7)

            tb_writer.add_scalar('loss/cls', rpn_cls_loss, tb_index)
            tb_writer.add_scalar('loss/loc', rpn_loc_loss, tb_index)
            tb_writer.add_scalar('loss/kp_hp_loss', kp_hp_loss, tb_index)
            tb_writer.add_scalar('loss/kp_hm_hp_loss', kp_hm_hp_loss, tb_index)
            tb_writer.add_scalar('loss/kp_hp_offset_loss', kp_hp_offset_loss, tb_index)
            # tb_writer.add_scalar('loss/kp', kp_loss, tb_index)
            end = time.time()

            if (iter + 1) % args.print_freq == 0:
                logger.info('Epoch: [{0}][{1}/{2}] lr: {lr:.6f}\t{batch_time:s}\t{data_time:s}'
                            '\t{rpn_cls_loss:s}\t{rpn_loc_loss:s}'
                            '\t{kp_hp_loss:s}\t{kp_hm_hp_loss:s}\t{kp_hp_offset_loss:s}'
                            '\t{kp_loss:s}\t{siammask_loss:s}'.format(
                            epoch+1, (iter + 1) % num_per_epoch, num_per_epoch, lr=cur_lr, batch_time=avg.batch_time,
                            data_time=avg.data_time, rpn_cls_loss=avg.rpn_cls_loss, rpn_loc_loss=avg.rpn_loc_loss,
                            kp_hp_loss=avg.kp_hp_loss, kp_hm_hp_loss=avg.kp_hm_hp_loss, kp_hp_offset_loss=avg.kp_hp_offset_loss,
                            kp_loss=avg.kp_loss, siammask_loss=avg.siammask_loss,))
                            # mask_iou_mean=avg.mask_iou_mean,
                            # mask_iou_at_5=avg.mask_iou_at_5,mask_iou_at_7=avg.mask_iou_at_7))
                print_speed(iter + 1, avg.batch_time.avg, args.epochs * num_per_epoch)


def save_checkpoint(state, is_best, filename='checkpoint.pth', best_file='model_best.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_file)

def args_process(opt):
    opt.reg_bbox = not opt.not_reg_bbox
    opt.hm_hp = not opt.not_hm_hp
    opt.reg_hp_offset = (not opt.not_reg_hp_offset) and opt.hm_hp
    return opt


if __name__ == '__main__':
    main()
