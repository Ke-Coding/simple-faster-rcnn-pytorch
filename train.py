import os
os.chdir('/content/drive/My Drive/lq_det_hyper/lq_det')


%reload_ext autoreload
%autoreload 2

import warnings
warnings.filterwarnings("ignore")   # no warn for torch1.3 about non-static forward



# from __future__ import absolute_import
# though cupy is not used but without this line, it raise errors...
import cupy as cp
import os

# import ipdb
import matplotlib
from tqdm import tqdm

import torch as tc
from utils.config import Config
from data.dataset import TrainDataset, ValDataset, inverse_normalize
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc
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
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
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
    losses = []
    # rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss, total_loss
    for epoch in range(Config.epoch):
        trainer.reset_meters()
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(train_loader)):
            scale = at.toscalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            # (rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss, total_loss)
            losses.append(trainer.train_step(img, bbox, label, scale))
            
            # if (ii + 1) % Config.plt_freq == 0:
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
        log_info = f"[ep{epoch}] best eval map:{best_map:.3g}, lr:{lr_:.4g}, eval map:{eval_result['map']:.3g}, avg tr loss:{trainer.get_meter_data()}"
        if trainer.vis:
            trainer.vis.log(log_info)
        else:
            print(log_info)


train()


import matplotlib.pyplot as plt
plt.show()
