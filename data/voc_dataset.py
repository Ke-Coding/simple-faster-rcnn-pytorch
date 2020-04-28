import os
# import xml.etree.ElementTree as ET

import json
import numpy as np

from .util import read_image

#
#
# class VOCBboxDataset:
#     """Bounding box dataset for PASCAL `VOC`_.
#
#     .. _`VOC`: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/
#
#     The index corresponds to each image.
#
#     When queried by an index, if :obj:`return_difficult == False`,
#     this dataset returns a corresponding
#     :obj:`img, bbox, label`, a tuple of an image, bounding boxes and labels.
#     This is the default behaviour.
#     If :obj:`return_difficult == True`, this dataset returns corresponding
#     :obj:`img, bbox, label, difficult`. :obj:`difficult` is a boolean array
#     that indicates whether bounding boxes are labeled as difficult or not.
#
#     The bounding boxes are packed into a two dimensional tensor of shape
#     :math:`(R, 4)`, where :math:`R` is the number of bounding boxes in
#     the image. The second axis represents attributes of the bounding box.
#     They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`, where the
#     four attributes are coordinates of the top left and the bottom right
#     vertices.
#
#     The labels are packed into a one dimensional tensor of shape :math:`(R,)`.
#     :math:`R` is the number of bounding boxes in the image.
#     The class name of the label :math:`l` is :math:`l` th element of
#     :obj:`VOC_BBOX_LABEL_NAMES`.
#
#     The array :obj:`difficult` is a one dimensional boolean array of shape
#     :math:`(R,)`. :math:`R` is the number of bounding boxes in the image.
#     If :obj:`use_difficult` is :obj:`False`, this array is
#     a boolean array with all :obj:`False`.
#
#     The type of the image, the bounding boxes and the labels are as follows.
#
#     * :obj:`img.dtype == numpy.float32`
#     * :obj:`bbox.dtype == numpy.float32`
#     * :obj:`label.dtype == numpy.int32`
#     * :obj:`difficult.dtype == numpy.bool`
#
#     Args:
#         data_dir (string): Path to the root of the training data.
#             i.e. "/data/image/voc/VOCdevkit/VOC2007/"
#         split ({'train', 'val', 'trainval', 'test'}): Select a split of the
#             dataset. :obj:`test` split is only available for
#             2007 dataset.
#         year ({'2007', '2012'}): Use a dataset prepared for a challenge
#             held in :obj:`year`.
#         use_difficult (bool): If :obj:`True`, use images that are labeled as
#             difficult in the original annotation.
#         return_difficult (bool): If :obj:`True`, this dataset returns
#             a boolean array
#             that indicates whether bounding boxes are labeled as difficult
#             or not. The default value is :obj:`False`.
#
#     """
#
#     def __init__(self, data_dir, split='trainval',
#                  use_difficult=False, return_difficult=False,
#                  ):
#
#         # if split not in ['train', 'trainval', 'val']:
#         #     if not (split == 'test' and year == '2007'):
#         #         warnings.warn(
#         #             'please pick split from \'train\', \'trainval\', \'val\''
#         #             'for 2012 dataset. For 2007 dataset, you can pick \'test\''
#         #             ' in addition to the above mentioned splits.'
#         #         )
#         id_list_file = os.path.join(
#             data_dir, 'ImageSets/Main/{0}.txt'.format(split))
#
#         self.ids = [id_.strip() for id_ in open(id_list_file)]
#         self.data_dir = data_dir
#         self.use_difficult = use_difficult
#         self.return_difficult = return_difficult
#         self.label_names = VOC_BBOX_LABEL_NAMES
#
#     def __len__(self):
#         return len(self.ids)
#
#     def get_example(self, i):
#         """Returns the i-th example.
#
#         Returns a color image and bounding boxes. The image is in CHW format.
#         The returned image is RGB.
#
#         Args:
#             i (int): The index of the example.
#
#         Returns:
#             tuple of an image and bounding boxes
#
#         """
#         id_ = self.ids[i]
#         anno = ET.parse(
#             os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))
#         bbox = list()
#         label = list()
#         difficult = list()
#         for obj in anno.findall('object'):
#             # when in not using difficult split, and the object is
#             # difficult, skipt it.
#             if not self.use_difficult and int(obj.find('difficult').text) == 1:
#                 continue
#
#             difficult.append(int(obj.find('difficult').text))
#             bndbox_anno = obj.find('bndbox')
#             # subtract 1 to make pixel indexes 0-based
#             bbox.append([
#                 int(bndbox_anno.find(tag).text) - 1
#                 for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
#             name = obj.find('name').text.lower().strip()
#             label.append(VOC_BBOX_LABEL_NAMES.index(name))
#         bbox = np.stack(bbox).astype(np.float32)
#         label = np.stack(label).astype(np.int32)
#         # When `use_difficult==False`, all elements in `difficult` are False.
#         difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)  # PyTorch don't support np.bool
#
#         # Load a image
#         img_file = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
#         img = read_image(img_file, color=True)
#
#         # if self.return_difficult:
#         #     return img, bbox, label, difficult
#         return img, bbox, label, difficult
#
#     __getitem__ = get_example
#
#
# VOC_BBOX_LABEL_NAMES = (
#     'aeroplane',
#     'bicycle',
#     'bird',
#     'boat',
#     'bottle',
#     'bus',
#     'car',
#     'cat',
#     'chair',
#     'cow',
#     'diningtable',
#     'dog',
#     'horse',
#     'motorbike',
#     'person',
#     'pottedplant',
#     'sheep',
#     'sofa',
#     'train',
#     'tvmonitor')
#


class LQCarDataSet:
    """Bounding box dataset for liqing's cars.
    * :obj:`img.dtype == numpy.float32`
    * :obj:`bbox.dtype == numpy.float32`
    * :obj:`label.dtype == numpy.int32`
    * :obj:`difficult.dtype == numpy.bool`
    Args:
        data_dir (string): Path to the root of the dataset.
            it should contain `imgs', `train_meta.json' and `val_meta.json'
        split ({'train', 'val'}): Select a split of the
            dataset. :obj:`test` split is only available for
            2007 dataset.
    
    Meta Example:
        {
            "filename": "003039.jpg",
            "image_height": 600,
            "image_width": 800,
            "instances": [{
                    "bbox": [213, 238, 433, 451],   # pixel start from 1, not 0
                    "label": 0
                }, {
                    "bbox": [458, 262, 627, 407],   # pixel start from 1, not 0
                    "label": 0
                }
            ]
        }
    """
    
    BBOX_LABEL_NAMES = ('car', )
    
    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.meta_list = None
        with open(os.path.join(self.data_dir, f'{split}_meta.json'), 'r') as f:
            self.meta_list = json.load(f)
        
    def __len__(self):
        return len(self.meta_list)
    
    def get_example(self, i):
        """Returns the i-th example.

        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and bounding boxes

        """
        meta = self.meta_list[i]
        
        bboxes = list()     # order: ymin, xmin, ymax, xmax
        labels = list()
        difficults = list()

        img = read_image(os.path.join(self.data_dir, 'imgs', meta['filename']), color=True)
        for instance in meta['instances']:
            label_idx, bbox = instance['label'], instance['bbox']
            xmin, ymin, xmax, ymax = bbox
            bboxes.append([ymin, xmin, ymax, xmax])
            labels.append(instance['label'])
            difficults.append(False)

        bboxes = np.stack(bboxes).astype(np.float32)
        labels = np.stack(labels).astype(np.int32)
        difficults = np.array(difficults, dtype=np.bool).astype(np.uint8)
        
        return img, bboxes, labels, difficults
    
    __getitem__ = get_example

