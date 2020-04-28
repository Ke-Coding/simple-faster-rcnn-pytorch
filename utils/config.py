from pprint import pprint


# Default Configs for training
# NOTE that, config items could be overwriten by passing argument through command line.
# e.g. --voc-data-dir='./data/'

class Config:
    # data
    # voc_data_dir = '/home/cy/.chainer/dataset/pfnet/chainercv/voc/VOCdevkit/VOC2007/'
    det_dataset_dir = '/content/drive/My Drive/datasets/det/lq_car_det'
    num_classes_except_bg = 1
    num_classes_include_bg = num_classes_except_bg + 1
    min_size = 600  # image resize
    max_size = 1000  # image resize
    batch_size = 1
    num_workers = 2
    test_num_workers = 2
    
    # sigma for l1_smooth_loss
    rpn_sigma = 3.
    roi_sigma = 1.
    
    # param for optimizer
    # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
    weight_decay = 0.0005
    # lr_decay = 0.1  # 1e-3 -> 1e-4
    warm_up_lr = 7.5e-4
    lr = warm_up_lr / 4
    warm_up_iter = 350
    warm_up_delta = (warm_up_lr - lr) / warm_up_iter
    
    # visualization
    env = 'faster-rcnn'  # visdom env
    port = 8097
    plt_freq = 40  # vis every N iter
    prt_freq = 128  # vis every N iter
    
    # preset
    data = 'voc'
    pretrained_model = 'vgg16'
    
    # training
    epoch = 10
    step_epochs = [3, 6]    # 0123 456 789
    step_decays = [0.1, 0.15, 0.2]
    sgd_nestrov = True
    sgd_momentum = 0.9
    
    use_adam = False  # Use Adam optimizer
    use_chainer = False  # try match everything as chainer
    use_drop = False  # use dropout in RoIHead
    # debug
    debug_file = '/tmp/debugf'
    
    # ckpts
    frc_ckpt_path = None
    # frc_ckpt_path = '/content/drive/My Drive/lq_det_hyper/lq_det/ckpt/fasterrcnn_12222105_0.712649824453_caffe_pretrain.pth'
    
    caffe_vgg = False  # use caffe pretrained model instead of torchvision
    caffe_vgg_path = None
    # caffe_vgg_path = '/content/drive/My Drive/lq_det_hyper/lq_det/ckpt/vgg16_caffe.pth'
    torchvision_vgg_path = '/content/drive/My Drive/lq_det_hyper/lq_det/ckpt/vgg16_torchvision.pth'
    
    @classmethod
    def _parse(cls, kwargs):
        state_dict = cls._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(cls, k, v)
        
        print('======user config========')
        pprint(cls._state_dict())
        print('==========end============')
    
    @classmethod
    def _state_dict(cls):
        return {k: getattr(cls, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}
