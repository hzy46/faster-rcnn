# coding: utf-8
import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
from voc_eval import voc_eval
from fast_rcnn.config import cfg
import pdb
import json


class sz(imdb):
    def __init__(self, image_set, path=None):
        imdb.__init__(self, 'sz_' + image_set)
        self._image_set = image_set
        self._path = self._get_default_path() if path is None \
            else path
        """神州的所有训练图片在training/training里面"""
        self._img_path = os.path.join(self._path, 'all_pic/')
        self._label_file = os.path.join(self._path, 'label.idl')
        self._classes = ('__background__',  # always index 0
                         'vehicle', 'pedestrian', 'cyclist', 'traffic_lights')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'sz')

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        image_set_file = os.path.join(self._path, self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        # cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        # if os.path.exists(cache_file):
        #     with open(cache_file, 'rb') as fid:
        #         roidb = cPickle.load(fid)
        #     print '{} gt roidb loaded from {}'.format(self.name, cache_file)
        #     return roidb

        gt_roidb = self.load_gt_roidb()
        # with open(cache_file, 'wb') as fid:
        #     cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        # print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def load_gt_roidb(self):
        with open(self._label_file) as f:
            result_dict = {}
            for line in f:
                j = json.loads(line)
                key = j.keys()[0]
                img_index = key.split('.')[0]
                bbox_list = j[key]
                num_objs = len(bbox_list)
                boxes = np.zeros((num_objs, 4), dtype=np.uint16)
                gt_classes = np.zeros((num_objs), dtype=np.int32)
                overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
                seg_areas = np.zeros((num_objs), dtype=np.float32)
                for ix, box in enumerate(bbox_list):
                    """神州这里的标记框本身是从0开始的"""
                    x1 = box[0]
                    y1 = box[1]
                    x2 = box[2]
                    y2 = box[3]
                    """类别需要把20（信号灯）转成4"""
                    cls = box[4]
                    if cls == 20:
                        cls = 4

                    boxes[ix, :] = [x1, y1, x2, y2]
                    gt_classes[ix] = cls
                    overlaps[ix, cls] = 1.0
                    seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
                overlaps = scipy.sparse.csr_matrix(overlaps)
                result_dict[img_index] = {'boxes': boxes,
                                          'gt_classes': gt_classes,
                                          'gt_overlaps': overlaps,
                                          'flipped': False,
                                          'seg_areas': seg_areas}
        """dict -> list"""
        result_list = []
        for image_index in self._image_index:
            result_list.append(result_dict[image_index])

        return result_list

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._img_path,
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

if __name__ == '__main__':
    from datasets.sz import sz
    d = sz('train')
    res = d.roidb
    print(len(res))
    from IPython import embed
    embed()
