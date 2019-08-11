import argparse
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES']='0, 1, 3, 4'
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
from utils.image import get_max_preds, save_batch_heatmaps

from datasets.siam_rcnn_dataset import DataSets
from utils.lr_helper import build_lr_scheduler
from tensorboardX import SummaryWriter

from utils.config_helper import load_config
from torch.utils.collect_env import get_pretty_env_info
from torchvision.transforms import ToTensor

torch.backends.cudnn.benchmark = True

best_acc = 0.
tb_index = 0
tb_val_index = 0
parser = argparse.ArgumentParser(description='PyTorch Tracking SiamMask Training')

parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--save_freq', default=5, type=int,
                    help='save frequency of model')
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
parser.add_argument('--output_size', default=56, type=int,
                    help='the output size of pose or mask branch')
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
parser.add_argument('--debug', action='store_true')



def select_pred_heatmap(p_m, weight, o_sz=63, g_sz=127):

    weight = weight.view(-1)
    pos = Variable(weight.data.eq(1).nonzero().squeeze())
    if pos.nelement() == 0: return p_m.sum() * 0

    if len(p_m.shape) == 4:
        # p_m = p_m.permute(0, 2, 3, 1).contiguous().view(-1, 17, o_sz, o_sz)
        p_m = torch.index_select(p_m, 0, pos)
        # p_m = nn.UpsamplingBilinear2d(size=[g_sz, g_sz])(p_m)
    else:
        p_m = torch.index_select(p_m, 0, pos)
        p_m = p_m.view(-1, 17, g_sz, g_sz)
    return p_m

def select_gt_img(mask, weight, channel=3, g_sz=127):

    weight = weight.view(-1)
    pos = Variable(weight.data.eq(1).nonzero().squeeze())
    if pos.nelement() == 0: return mask.sum() * 0

    mask_uf = F.unfold(mask, (g_sz, g_sz), padding=32, stride=8)
    mask_uf = torch.transpose(mask_uf, 1, 2).contiguous().view(-1, channel, g_sz * g_sz)
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
    train_set = DataSets(cfg['train_datasets'], cfg['anchors'])
    train_set.shuffle()

    logger.info("build val dataset")  # val_dataset
    if not 'val_datasets' in cfg.keys():
        cfg['val_datasets'] = cfg['train_datasets']
    val_set = DataSets(cfg['val_datasets'], cfg['anchors'])
    val_set.shuffle()

    train_loader = DataLoader(train_set, shuffle=True, batch_size=args.batch, num_workers=args.workers,
                              pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_set, shuffle=True, batch_size=args.batch, num_workers=args.workers,
                            pin_memory=False, drop_last=True)

    logger.info('build dataset done')
    return train_loader, val_loader


def build_opt_lr(model, cfg, args, epoch):
    backbone_feature = model.features.param_groups(cfg['lr']['start_lr'], cfg['lr']['feature_lr_mult'])
    if len(backbone_feature) == 0:
        trainable_params = model.rpn_model.param_groups(cfg['lr']['start_lr'], cfg['lr']['rpn_lr_mult'], 'mask')
    else:
        trainable_params = backbone_feature + \
                           model.rpn_model.param_groups(cfg['lr']['start_lr'], cfg['lr']['rpn_lr_mult']) + \
                           model.kp_model.param_groups(cfg['lr']['start_lr'], cfg['lr']['mask_lr_mult']) + \
                           model.pose_corr.param_groups(cfg['lr']['start_lr'], cfg['lr']['mask_lr_mult'])

    optimizer = torch.optim.SGD(trainable_params, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = build_lr_scheduler(optimizer, cfg['lr'], epochs=args.epochs)

    lr_scheduler.step(epoch)

    return optimizer, lr_scheduler

def is_valid_number(x):
        return not(math.isnan(x) or math.isinf(x) or x > 1e4)

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
        model = Custom(pretrain=True, opts=args,
                       anchors=train_loader.dataset.anchors)
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
    global cur_lr

    if not os.path.exists(args.save_dir):  # makedir/save model
        os.makedirs(args.save_dir)
    num_per_epoch = len(train_loader.dataset) // args.batch
    num_per_epoch_val = len(val_loader.dataset) // args.batch

    for epoch in range(args.start_epoch, args.epochs):
        lr_scheduler.step(epoch)
        cur_lr = lr_scheduler.get_cur_lr()
        logger = logging.getLogger('global')
        train_avg = AverageMeter()
        val_avg = AverageMeter()
        
        if dist_model.module.features.unfix(epoch/args.epochs):
            logger.info('unfix part model.')
            optimizer, lr_scheduler = build_opt_lr(dist_model.module, cfg, args, epoch)

        train(train_loader, dist_model, optimizer, lr_scheduler, epoch, cfg, train_avg, num_per_epoch)

        if dist_model.module.features.unfix(epoch/args.epochs):
            logger.info('unfix part model.')
            optimizer, lr_scheduler = build_opt_lr(dist_model.module, cfg, args, epoch)

        if (epoch+1) % args.save_freq == 0:
            save_checkpoint({
                    'epoch': epoch,
                    'arch': args.arch,
                    'state_dict': dist_model.module.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                    'anchor_cfg': cfg['anchors']
                }, False,
                os.path.join(args.save_dir, 'checkpoint_e%d.pth' % (epoch)),
                os.path.join(args.save_dir, 'best.pth'))

            validation(val_loader, dist_model, epoch, cfg, val_avg, num_per_epoch_val)


def BNtoFixed(m):
    class_name = m.__class__.__name__
    if class_name.find('BatchNorm') != -1:
        m.eval()


# train one epoch
def train(train_loader, model, optimizer, lr_scheduler, epoch, cfg, avg, num_per_epoch):
    global tb_index, best_acc, cur_lr, logger
    end = time.time()
    cur_lr = lr_scheduler.get_cur_lr()
    model.train()
    model = model.cuda()
    
    # model.module.rpn_model.eval()
    # model.module.kp_model.eval()

    logger.info('train epoch:{}'.format(epoch))

    for iter, input in enumerate(train_loader):
        tb_index += iter
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
        #gpu_profile(frame=sys._getframe(), event='line', arg=None)
          
        outputs = model(x_rpn, x_kp)

        rpn_cls_loss, rpn_loc_loss, kp_losses = torch.mean(outputs['losses'][0]),\
                                                    torch.mean(outputs['losses'][1]),\
                                                    outputs['losses'][3]
        kp_loss = torch.mean(kp_losses['loss'])
        kp_hp_loss = torch.mean(kp_losses['hp_loss'])
        kp_hm_hp_loss = torch.mean(kp_losses['hm_hp_loss'])
        kp_hp_offset_loss = torch.mean(kp_losses['hp_offset_loss'])

        # mask_iou_mean, mask_iou_at_5, mask_iou_at_7 = torch.mean(outputs['accuracy'][0]), torch.mean(outputs['accuracy'][1]), torch.mean(outputs['accuracy'][2])
        kp_avg_acc = torch.mean(outputs['accuracy'][1])

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
        else:
            print('not valid loss')
        # gpu_profile(frame=sys._getframe(), event='line', arg=None)

        siammask_loss = loss.item()

        batch_time = time.time() - end

        avg.update(batch_time=batch_time, rpn_cls_loss=rpn_cls_loss, rpn_loc_loss=rpn_loc_loss,
                   kp_hp_loss=kp_hp_loss, kp_hm_hp_loss=kp_hm_hp_loss, kp_hp_offset_loss=kp_hp_offset_loss,
                   kp_loss=kp_loss, siammask_loss=siammask_loss, kp_avg_acc=kp_avg_acc)
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
                        '\t{kp_loss:s}\t{siammask_loss:s}'
                        '\t{kp_avg_acc:s}'.format(
                        epoch+1, (iter + 1) % num_per_epoch, num_per_epoch, lr=cur_lr, batch_time=avg.batch_time,
                        data_time=avg.data_time, rpn_cls_loss=avg.rpn_cls_loss, rpn_loc_loss=avg.rpn_loc_loss,
                        kp_hp_loss=avg.kp_hp_loss, kp_hm_hp_loss=avg.kp_hm_hp_loss, kp_hp_offset_loss=avg.kp_hp_offset_loss,
                        kp_loss=avg.kp_loss, siammask_loss=avg.siammask_loss, kp_avg_acc=avg.kp_avg_acc))
                        # mask_iou_mean=avg.mask_iou_mean,
                        # mask_iou_at_5=avg.mask_iou_at_5,mask_iou_at_7=avg.mask_iou_at_7))
            print_speed(num_per_epoch * epoch + iter + 1, avg.batch_time.avg, args.epochs * num_per_epoch)


def validation(val_loader, model, epoch, cfg, avg, num_per_epoch_val):
    global tb_val_index, best_acc, logger
    end = time.time()
    model.eval()
    model = model.cuda()

    logger.info('val epoch:{}'.format(epoch))
    with torch.no_grad():
        for iter, input in enumerate(val_loader):
            tb_val_index += iter

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
            x_rpn['anchors'] = val_loader.dataset.anchors.all_anchors[0]

            outputs = model(x_rpn, x_kp)

            rpn_cls_loss, rpn_loc_loss, kp_losses = torch.mean(outputs['losses'][0]),\
                                                        torch.mean(outputs['losses'][1]),\
                                                        outputs['losses'][3]
            kp_loss = torch.mean(kp_losses['loss'])
            kp_hp_loss = torch.mean(kp_losses['hp_loss'])
            kp_hm_hp_loss = torch.mean(kp_losses['hm_hp_loss'])
            kp_hp_offset_loss = torch.mean(kp_losses['hp_offset_loss'])

            # mask_iou_mean, mask_iou_at_5, mask_iou_at_7 = torch.mean(outputs['accuracy'][0]), torch.mean(outputs['accuracy'][1]), torch.mean(outputs['accuracy'][2])
            kp_avg_acc = torch.mean(outputs['accuracy'][1])

            cls_weight, reg_weight, kp_weight = cfg['loss']['weight']

            loss = rpn_cls_loss * cls_weight + rpn_loc_loss * reg_weight + kp_loss * kp_weight
            siammask_loss = loss.item()

            batch_time = time.time() - end

            avg.update(batch_time=batch_time, rpn_cls_loss=rpn_cls_loss, rpn_loc_loss=rpn_loc_loss,
                       kp_hp_loss=kp_hp_loss, kp_hm_hp_loss=kp_hm_hp_loss, kp_hp_offset_loss=kp_hp_offset_loss,
                       kp_loss=kp_loss, siammask_loss=siammask_loss, kp_avg_acc=kp_avg_acc)
                       # mask_iou_mean=mask_iou_mean, mask_iou_at_5=mask_iou_at_5, mask_iou_at_7=mask_iou_at_7)

            tb_writer.add_scalar('val_loss/cls', rpn_cls_loss, tb_val_index)
            tb_writer.add_scalar('val_loss/loc', rpn_loc_loss, tb_val_index)
            tb_writer.add_scalar('val_loss/kp_hp_loss', kp_hp_loss, tb_val_index)
            tb_writer.add_scalar('val_loss/kp_hm_hp_loss', kp_hm_hp_loss, tb_val_index)
            tb_writer.add_scalar('val_loss/kp_hp_offset_loss', kp_hp_offset_loss, tb_val_index)
            # tb_writer.add_scalar('loss/kp', kp_loss, tb_index)
            end = time.time()

            if (iter + 1) % args.print_freq == 0:
                logger.info('Epoch: [{0}][{1}/{2}] Validation:\t{batch_time:s}\t{data_time:s}'
                            '\t{rpn_cls_loss:s}\t{rpn_loc_loss:s}'
                            '\t{kp_hp_loss:s}\t{kp_hm_hp_loss:s}\t{kp_hp_offset_loss:s}'
                            '\t{kp_loss:s}\t{siammask_loss:s}'
                            '\t{kp_avg_acc:s}'.format(
                            epoch+1, (iter + 1) % num_per_epoch_val, num_per_epoch_val, batch_time=avg.batch_time,
                            data_time=avg.data_time, rpn_cls_loss=avg.rpn_cls_loss, rpn_loc_loss=avg.rpn_loc_loss,
                            kp_hp_loss=avg.kp_hp_loss, kp_hm_hp_loss=avg.kp_hm_hp_loss, kp_hp_offset_loss=avg.kp_hp_offset_loss,
                            kp_loss=avg.kp_loss, siammask_loss=avg.siammask_loss, kp_avg_acc=avg.kp_avg_acc))
                            # mask_iou_mean=avg.mask_iou_mean,
                            # mask_iou_at_5=avg.mask_iou_at_5,mask_iou_at_7=avg.mask_iou_at_7))
                # print_speed(iter + 1, avg.batch_time.avg, args.epochs * num_per_epoch_val)


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
