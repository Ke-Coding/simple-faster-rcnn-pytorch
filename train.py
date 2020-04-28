import os
import time
os.chdir('/content/drive/My Drive/lq_det_hyper/lq_det')

TIME_STR = time.strftime('%Y-%m-%d-%H:%M:%S')
EXP_DIR = os.path.join(os.getcwd(), 'exp', TIME_STR)

%reload_ext autoreload
%autoreload 2

import warnings
warnings.filterwarnings("ignore")   # no warn for torch1.3 about non-static forward



# from __future__ import absolute_import
# though cupy is not used but without this line, it raise errors...
import cupy as cp
import os

# import ipdb
from tqdm import tqdm

import torch as tc
import datetime
from utils.config import Config
from data.dataset import TrainDataset, ValDataset, inverse_normalize
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc
import matplotlib
%matplotlib inline

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource


rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

# matplotlib.use('agg')


def eval(dataloader, faster_rcnn):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for it, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in enumerate(dataloader):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes.extend(gt_bboxes_.numpy().tolist())
        gt_labels.extend(gt_labels_.numpy().tolist())
        gt_difficults.extend(gt_difficults_.numpy().tolist())
        pred_bboxes.extend(pred_bboxes_)
        pred_labels.extend(pred_labels_)
        pred_scores.extend(pred_scores_)
    
    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result


def train(**kwargs):
    Config._parse(kwargs)
    
    train_dataset = TrainDataset(Config)
    train_loader = data_.DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        # pin_memory=True,
        num_workers=Config.num_workers
    )
    print('train data loaded')
    val_dataset = ValDataset(Config)
    val_loader = data_.DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        num_workers=Config.test_num_workers,
        shuffle=False,
        pin_memory=True
    )
    print('val data loaded')
    
    faster_rcnn = FasterRCNNVGG16(n_fg_class=Config.num_classes_except_bg)
    print('model construct completed')
    
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if Config.frc_ckpt_path:
        trainer.load(Config.frc_ckpt_path)
        print('load pretrained model from %s' % Config.frc_ckpt_path)
    if trainer.vis:
        trainer.vis.text(train_dataset.db.label_names, win='labels')
    
    best_map, best_path = 0, ''
    train_losses = []
    train_lrs = []
    val_maps = []
    
    log_file = open(os.path.join(EXP_DIR, 'log.txt'), 'w')

    # rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss, total_loss
    time_passed = 0
    last_t = time.time()
    for epoch in range(Config.epoch):
        trainer.reset_meters()
        tot_it = len(train_loader)
        for it, (img, bbox_, label_, scale) in enumerate(train_loader):
            scale = at.toscalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            # (rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss, total_loss)
            train_loss = trainer.train_step(img, bbox, label, scale)
            train_losses.append(train_loss)
            
            # rpn_loc_loss
            # rpn_cls_loss
            # roi_loc_loss
            # roi_cls_loss
            if it % Config.prt_freq == 0 or it == tot_it - 1:
                avg_speed = time_passed / it if it != 0 else 0
                remain_secs = (tot_it - it - 1) * avg_speed + tot_it * (Config.epoch - epoch - 1) * avg_speed
                remain_time = datetime.timedelta(seconds=round(remain_secs))
                finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remain_secs))
                log_str = (
                    f'ep[{epoch}/{Config.epoch}], it[{it + 1}/{tot_it}]:'
                    f' loc0[{train_loss.rpn_loc_loss:.4g}],'
                    f' cls0[{train_loss.rpn_cls_loss:.4g}],'
                    f' loc1[{train_loss.roi_loc_loss:.4g}],'
                    f' cls1[{train_loss.roi_cls_loss:.4g}],'
                    f' eta[{remain_time}] ({finish_time})'
                )
                print(log_str)
                print(log_str, file=log_file)
            
            # if (it + 1) % Config.plt_freq == 0:
            #     # if os.path.exists(Config.debug_file):
            #     #     ipdb.set_trace()
            #
            #     # plot loss
            #     trainer.vis.plot_many(trainer.get_meter_data())
            #
            #     # plot groud truth bboxes
            #     ori_img_ = inverse_normalize(at.tonumpy(img[0]))
            #     gt_img = visdom_bbox(ori_img_,
            #                          at.tonumpy(bbox_[0]),
            #                          at.tonumpy(label_[0]))
            #     trainer.vis.img('gt_img', gt_img)
            #
            #     # plot predicti bboxes
            #     _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
            #     pred_img = visdom_bbox(ori_img_,
            #                            at.tonumpy(_bboxes[0]),
            #                            at.tonumpy(_labels[0]).reshape(-1),
            #                            at.tonumpy(_scores[0]))
            #     trainer.vis.img('pred_img', pred_img)
            #
            #     # rpn confusion matrix(meter)
            #     trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
            #     # roi confusion matrix
            #     trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())
        
        eval_result = eval(val_loader, faster_rcnn)
        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map)
        if epoch in Config.step_epochs:
            # trainer.load(best_path)
            lr_decay = Config.step_decays[Config.step_epochs.index(epoch)]
            trainer.faster_rcnn.scale_lr(lr_decay)
        
        # plot details
        if trainer.vis:
            trainer.vis.plot('test_map', eval_result['map'])
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        train_lrs.append(lr_)
        val_maps.append(eval_result['map'])
        log_info = f"\n ==> [ep{epoch}] best eval map:{best_map:.3g}, lr:{lr_:.4g}, eval map:{eval_result['map']:.3g}, avg tr loss:{trainer.get_meter_data()}\n"
        if trainer.vis:
            trainer.vis.log(log_info)
        else:
            print(log_info)
            print(log_info, file=log_file)

        time_passed += last_t - time.time()
        last_t = time.time()

    # rpn_loc_loss
    # rpn_cls_loss
    # roi_loc_loss
    # roi_cls_loss
    # total_loss

    log_file.close()
    
    total_iters = len(train_losses)
    plt.figure()
    plt.subplot(2, 4, 1)
    plt.tight_layout(pad=2.5)
    plt.plot(list(range(total_iters)), [l.rpn_loc_loss for l in train_losses],
             label='rpn_loc_loss', c='green')
    plt.xlabel('iter')
    plt.ylabel('rpn_loc_loss')
    plt.legend(loc='lower right')

    plt.subplot(2, 4, 2)
    plt.tight_layout(pad=2.5)
    plt.plot(list(range(total_iters)), [l.rpn_cls_loss for l in train_losses],
             label='rpn_cls_loss', c='steelblue')
    plt.xlabel('iter')
    plt.ylabel('rpn_cls_loss')
    plt.legend(loc='lower right')

    plt.subplot(2, 4, 3)
    plt.tight_layout(pad=2.5)
    plt.plot(list(range(total_iters)), [l.roi_loc_loss for l in train_losses],
             label='roi_loc_loss', c='darkviolet')
    plt.xlabel('iter')
    plt.ylabel('roi_loc_loss')
    plt.legend(loc='lower right')

    plt.subplot(2, 4, 4)
    plt.tight_layout(pad=2.5)
    plt.plot(list(range(total_iters)), [l.roi_cls_loss for l in train_losses],
             label='roi_cls_loss', c='tomato')
    plt.xlabel('iter')
    plt.ylabel('roi_cls_loss')
    plt.legend(loc='lower right')

    plt.subplot(2, 4, 5)
    plt.tight_layout(pad=2.5)
    plt.plot(list(range(Config.epoch)), train_lrs,
             label='train lr', c='blue')
    plt.xlabel('epoch')
    plt.ylabel('lr')
    plt.legend(loc='lower right')

    plt.subplot(2, 4, 6)
    plt.tight_layout(pad=2.5)
    plt.plot(list(range(Config.epoch)), val_maps,
             label='val mAP', c='red')
    plt.xlabel('epoch')
    plt.ylabel('val mAP')
    plt.legend(loc='lower right')


train()


import matplotlib.pyplot as plt
plt.show()
