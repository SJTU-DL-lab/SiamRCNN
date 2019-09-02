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

from utils.log_helper import init_log, add_file_handler
from utils.load_helper import load_pretrain, restore_from
from utils.average_meter_helper import AverageMeter
from utils.image import save_gt_pred_heatmaps, save_batch_resized_heatmaps, get_max_preds_loc
from utils.pose_evaluate import accuracy, coco_eval

from datasets.siam_rcnn_val_dataset import DataSets
from utils.lr_helper import build_lr_scheduler

from utils.config_helper import load_config
from torch.utils.collect_env import get_pretty_env_info
from torchvision.transforms import ToTensor


torch.backends.cudnn.benchmark = True

best_acc = 0.
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


def collect_env_info():
    env_str = get_pretty_env_info()
    env_str += "\n        OpenCV ({})".format(cv2.__version__)
    return env_str


def build_data_loader(cfg):
    logger = logging.getLogger('global')

    logger.info("build train dataset")  # train_dataset
    train_set = DataSets(cfg['train_datasets'], cfg['anchors'])
    # train_set.shuffle()

    logger.info("build val dataset")  # val_dataset
    if not 'val_datasets' in cfg.keys():
        cfg['val_datasets'] = cfg['train_datasets']
    val_set = DataSets(cfg['val_datasets'], cfg['anchors'])
    # val_set.shuffle()

    train_loader = DataLoader(train_set, shuffle=True, batch_size=args.batch, num_workers=args.workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, shuffle=True, batch_size=args.batch, num_workers=args.workers,
                            pin_memory=True, drop_last=True)

    logger.info('build dataset done')
    return train_loader, val_loader

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
    else:
        raise Exception("Pretrained weights must be loaded!")

    model = model.cuda()
    dist_model = torch.nn.DataParallel(model, list(range(torch.cuda.device_count()))).cuda()

    logger.info('model prepare done')

    logger = logging.getLogger('global')
    val_avg = AverageMeter()

    validation(val_loader, dist_model, cfg, val_avg)

def BNtoFixed(m):
    class_name = m.__class__.__name__
    if class_name.find('BatchNorm') != -1:
        m.eval()

def validation(val_loader, model, cfg, avg):
    global tb_val_index, best_acc, logger
    num_per_epoch = len(val_loader.dataset)
    end = time.time()
    model.eval()
    valdata = []
    used_img_id = {}
    with torch.no_grad():
        for iter, input in enumerate(val_loader):
            # if iter > 100:
            #     break
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
                # 'label_mask': torch.autograd.Variable(input[6]).cuda()
            }
            x_kp = input[6]
            x_kp = {x: torch.autograd.Variable(y).cuda() for x, y in x_kp.items()}
            x_rpn['anchors'] = val_loader.dataset.anchors.all_anchors[0]

            outputs = model(x_rpn, x_kp)

            pred_kp = outputs['predict'][2][0]['hm_hp']
            kp_avg_acc = torch.mean(outputs['accuracy'][1])

            # print('pred kp shape: ', pred_kp.shape)
            # batch_img = x_rpn['search'].expand(x_kp['hm_hp'].size(0), -1, -1, -1)
            # gt_img, pred_img = save_gt_pred_heatmaps(batch_img, x_kp['hm_hp'], pred_kp, 'test_imgs/test_{}.jpg'.format(iter))
            rpn_cls_loss, rpn_loc_loss, kp_losses = torch.mean(outputs['losses'][0]),\
                                                               torch.mean(outputs['losses'][1]),\
                                                               outputs['losses'][3]
            kp_loss = torch.mean(kp_losses['loss'])
            kp_hp_loss = torch.mean(kp_losses['hp_loss'])
            kp_hm_hp_loss = torch.mean(kp_losses['hm_hp_loss'])
            kp_hp_offset_loss = torch.mean(kp_losses['hp_offset_loss'])

            cls_weight, reg_weight, kp_weight = cfg['loss']['weight']

            loss = rpn_cls_loss * cls_weight + rpn_loc_loss * reg_weight + kp_loss * kp_weight
            siammask_loss = loss.item()

            batch_time = time.time() - end

            offset_loc = x_kp['hp_offset'].cpu().detach().numpy()
            img_ids = input[9].cpu().detach().numpy()
            preds, maxvals = get_max_preds_loc(pred_kp.cpu().detach().numpy(), offset_loc)
            # preds = preds.astype(np.uint8)

            for i in range(preds.shape[0]):
                img_id = img_ids[i]
                temp_dict = dict()
                temp_dict["image_id"] = int(img_id)
                temp_dict["keypoints"] = preds[i].tolist()
                # print("============temp_dict[image_id] ========", temp_dict["image_id"])
                temp_dict["category_id"] = 1
                temp_dict["score"] = 0.7
                # print("============temp_dict[keypoints] ========", temp_dict["keypoints"])
                valdata.append(temp_dict)
                # img_list.append(int(img_ids[i]))
            # print(valdata)

            if args.debug:
                box_imgs, roi_imgs, hp_imgs = outputs['debug']
                # grid_img, resized_img = save_batch_resized_heatmaps(roi_imgs.transpose(1, 3),
                #                                                     hp_imgs, 'debug/feat_{}.jpg'.format(iter))
                # cv2.imwrite('debug/heatmap_{}.jpg'.format(iter), grid_img)
                # print('hp_imgs shape: ', hp_imgs.shape)
                grid_img, resized_img = save_batch_resized_heatmaps(hp_imgs,
                                                                    pred_kp, 'output/pred_hm_{}.jpg'.format(iter), out_skelet=True)
                cv2.imwrite('output/heatmap_{}.jpg'.format(iter), grid_img)
                # grid_img, resized_img = save_batch_resized_heatmaps(x_rpn['search'],
                #                                                     x_kp['hm_hp'], 'debug/feat_{}.jpg'.format(iter))
                # cv2.imwrite('debug/heatmap_{}.jpg'.format(iter), grid_img)
                box_imgs = box_imgs.transpose(1, 2).int().cpu().detach().numpy()
                roi_imgs = roi_imgs.transpose(1, 2).int().cpu().detach().numpy()
                for img_id in range(box_imgs.shape[0]):
                    cv2.imwrite('./debug/box_img{}_{}.png'.format(iter, img_id), box_imgs[img_id])
                for img_id in range(len(roi_imgs)):
                    cv2.imwrite('./debug/roi_img{}_{}.png'.format(iter, img_id), roi_imgs[img_id])

            avg.update(batch_time=batch_time, rpn_cls_loss=rpn_cls_loss, rpn_loc_loss=rpn_loc_loss,
                       kp_hp_loss=kp_hp_loss, kp_hm_hp_loss=kp_hm_hp_loss, kp_hp_offset_loss=kp_hp_offset_loss,
                       kp_loss=kp_loss, siammask_loss=siammask_loss, kp_avg_acc=kp_avg_acc)

            end = time.time()

            if (iter + 1) % args.print_freq == 0:
                logger.info('Validation: [{0}/{1}]\t{batch_time:s}\t{data_time:s}'
                            '\t{rpn_cls_loss:s}\t{rpn_loc_loss:s}'
                            '\t{kp_hp_loss:s}\t{kp_hm_hp_loss:s}\t{kp_hp_offset_loss:s}'
                            '\t{kp_loss:s}\t{siammask_loss:s}'
                            '\t{kp_avg_acc:s}'.format(
                                iter, num_per_epoch, batch_time=avg.batch_time, data_time=avg.data_time,
                                rpn_cls_loss=avg.rpn_cls_loss, rpn_loc_loss=avg.rpn_loc_loss,
                                kp_hp_loss=avg.kp_hp_loss, kp_hm_hp_loss=avg.kp_hm_hp_loss, kp_hp_offset_loss=avg.kp_hp_offset_loss,
                                kp_loss=avg.kp_loss, siammask_loss=avg.siammask_loss, kp_avg_acc=avg.kp_avg_acc))

    dt_file = './coco_eval/person_keypoints_val2017_result_new.json'
    gt_file = './coco_eval/person_keypoints_val2017_siampose_multi_new.json'
    json.dump(valdata, open(dt_file, 'w'), indent=4, sort_keys=True)
    coco_eval(gt_file, dt_file)

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
