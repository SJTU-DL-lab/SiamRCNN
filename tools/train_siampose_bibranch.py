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

from datasets.siam_pose_dataset import DataSets

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


best_acc = 0.

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals

def save_batch_heatmaps(batch_image, batch_heatmaps, file_name,
                        normalize=True, toTensor=ToTensor()):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros((batch_size*heatmap_height,
                           (num_joints+1)*heatmap_width,
                           3),
                          dtype=np.uint8)

    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            cv2.circle(resized_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap*0.7 + resized_image*0.3
            cv2.circle(masked_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j+1)
            width_end = heatmap_width * (j+2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image
        # print('brefore:', grid_image.shape)
        grid_image = copy.deepcopy(grid_image[:, :, ::-1])
        # print('after:', grid_image)
        out_image = toTensor(grid_image)


    return out_image

def select_pred_heatmap(p_m, weight, o_sz=63, g_sz=127):

    weight = weight.view(-1)
    pos = Variable(weight.data.eq(1).nonzero().squeeze())
    if pos.nelement() == 0: return p_m.sum() * 0

    p_m = p_m.permute(0, 2, 3, 1).contiguous().view(-1, 17, o_sz, o_sz)
    p_m = torch.index_select(p_m, 0, pos)
    p_m = nn.UpsamplingBilinear2d(size=[g_sz, g_sz])(p_m)
    return p_m

def select_gt_img(mask, weight, channel=3, g_sz=127):

    weight = weight.view(-1)
    pos = Variable(weight.data.eq(1).nonzero().squeeze())
    if pos.nelement() == 0: return mask.sum() * 0

    mask_uf = F.unfold(mask, (g_sz, g_sz), padding=32, stride=8)
    mask_uf = torch.transpose(mask_uf, 1, 2).contiguous().view(-1, g_sz * g_sz * channel)
    mask_uf = torch.index_select(mask_uf, 0, pos)
    mask_uf = mask_uf.view(-1, channel, g_sz, g_sz)

    return mask_uf

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
                           model.mask_model.param_groups(cfg['lr']['start_lr'], cfg['lr']['mask_lr_mult'])

    optimizer = torch.optim.SGD(trainable_params, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = build_lr_scheduler(optimizer, cfg['lr'], epochs=args.epochs)

    lr_scheduler.step(epoch)

    return optimizer, lr_scheduler


def main():
    global args, best_acc, tb_writer, logger
    args = parser.parse_args()

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

    if args.arch == 'Custom':
        from custom_double_branch import Custom
        model = Custom(pretrain=True, anchors=cfg['anchors'])
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


def train(train_loader, model, optimizer, lr_scheduler, epoch, cfg):
    global tb_index, best_acc, cur_lr, logger
    cur_lr = lr_scheduler.get_cur_lr()
    logger = logging.getLogger('global')
    avg = AverageMeter()
    model.module.features.eval()
    model.module.rpn_model.eval()
    model.module.mask_model.eval()
    # model.train()
    model = model.cuda()
    end = time.time()

    def is_valid_number(x):
        return not(math.isnan(x) or math.isinf(x) or x > 1e4)

    num_per_epoch = len(train_loader.dataset) // args.epochs // args.batch
    start_epoch = epoch
    epoch = epoch
    for iter, input in enumerate(train_loader):

        if epoch != iter // num_per_epoch + start_epoch:  # next epoch
            epoch = iter // num_per_epoch + start_epoch

            if not os.path.exists(args.save_dir):  # makedir/save model
                os.makedirs(args.save_dir)

            save_checkpoint({
                    'epoch': epoch,
                    'arch': args.arch,
                    'state_dict': model.module.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                    'anchor_cfg': cfg['anchors']
                }, False,
                os.path.join(args.save_dir, 'checkpoint_e%d.pth' % (epoch)),
                os.path.join(args.save_dir, 'best.pth'))

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
        x = {
            'cfg': cfg,
            'template': torch.autograd.Variable(input[0]).cuda(),
            'search': torch.autograd.Variable(input[1]).cuda(),
            'label_cls': torch.autograd.Variable(input[2]).cuda(),
            'label_loc': torch.autograd.Variable(input[3]).cuda(),
            'label_loc_weight': torch.autograd.Variable(input[4]).cuda(),
            'label_mask': torch.autograd.Variable(input[6]).cuda(),
            'label_kp_weight': torch.autograd.Variable(input[7]).cuda(),
            'label_mask_weight': torch.autograd.Variable(input[8]).cuda(),
            'label_kp': torch.autograd.Variable(input[9]).cuda()
        }

        outputs = model(x)

        rpn_cls_loss, rpn_loc_loss, rpn_kp_loss, rpn_heatmap_loss = torch.mean(outputs['losses'][0]),\
                                                                    torch.mean(outputs['losses'][1]),\
                                                                    torch.mean(outputs['losses'][2]),\
                                                                    torch.mean(outputs['losses'][3])

        # mask_iou_mean, mask_iou_at_5, mask_iou_at_7 = torch.mean(outputs['accuracy'][0]), torch.mean(outputs['accuracy'][1]), torch.mean(outputs['accuracy'][2])
        htmap_pred = outputs['predict'][-1]
        kp_pred = outputs['predict'][-2]
        htmap_pred = htmap_pred.squeeze(1)
        kp_pred = kp_pred.squeeze(1)
        # htmap_pred = htmap_pred.permute(0, 2, 3, 1)
        # kp_pred = kp_pred.permute(0, 2, 3, 1)
        htmap_pred = htmap_pred.cpu().numpy()
        kp_pred = kp_pred.cpu().detach().numpy()
        f, (ax1, ax2) = plt.subplots(1, 2)
        print(htmap_pred[0].shape)
        print(kp_pred[0].shape)
        ax1.imshow(htmap_pred[0])
        ax2.imshow(kp_pred[0])
        plt.show()

        cls_weight, reg_weight, kp_weight, heatmap_weight = cfg['loss']['weight']

        loss = rpn_cls_loss * cls_weight + rpn_loc_loss * reg_weight + rpn_kp_loss * kp_weight + rpn_heatmap_loss * heatmap_weight

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
                   rpn_kp_loss=rpn_kp_loss * kp_weight, rpn_heatmap_loss=rpn_heatmap_loss * heatmap_weight,
                   siammask_loss=siammask_loss)
                   # mask_iou_mean=mask_iou_mean, mask_iou_at_5=mask_iou_at_5, mask_iou_at_7=mask_iou_at_7)

        tb_writer.add_scalar('loss/cls', rpn_cls_loss, tb_index)
        tb_writer.add_scalar('loss/loc', rpn_loc_loss, tb_index)
        tb_writer.add_scalar('loss/kp_reg', rpn_kp_loss * kp_weight, tb_index)
        tb_writer.add_scalar('loss/heatmap', rpn_heatmap_loss * heatmap_weight, tb_index)
        # tb_writer.add_scalar('mask/mIoU', mask_iou_mean, tb_index)
        # tb_writer.add_scalar('mask/AP@.5', mask_iou_at_5, tb_index)
        # tb_writer.add_scalar('mask/AP@.7', mask_iou_at_7, tb_index)
        end = time.time()

        if (iter + 1) % args.print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}] lr: {lr:.6f}\t{batch_time:s}\t{data_time:s}'
                        '\t{rpn_cls_loss:s}\t{rpn_loc_loss:s}\t{rpn_kp_loss:s}\t{rpn_heatmap_loss:s}\t{siammask_loss:s}'.format(
                        epoch+1, (iter + 1) % num_per_epoch, num_per_epoch, lr=cur_lr, batch_time=avg.batch_time,
                        data_time=avg.data_time, rpn_cls_loss=avg.rpn_cls_loss, rpn_loc_loss=avg.rpn_loc_loss,
                        rpn_kp_loss=avg.rpn_kp_loss, rpn_heatmap_loss=avg.rpn_heatmap_loss, siammask_loss=avg.siammask_loss))
                        # mask_iou_mean=avg.mask_iou_mean,
                        # mask_iou_at_5=avg.mask_iou_at_5,mask_iou_at_7=avg.mask_iou_at_7))
            print_speed(iter + 1, avg.batch_time.avg, args.epochs * num_per_epoch)


def save_checkpoint(state, is_best, filename='checkpoint.pth', best_file='model_best.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_file)


if __name__ == '__main__':
    main()
