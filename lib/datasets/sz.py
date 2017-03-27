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
from voc_eval import sz_eval
from fast_rcnn.config import cfg
import pdb
import json
import logging


class sz(imdb):
    def __init__(self, image_set, path=None):
        imdb.__init__(self, 'sz_' + image_set)
        self._image_set = image_set
        self._path = self._get_default_path() if path is None \
            else path
        self._img_path = os.path.join(self._path, 'all_pic/')
        self._label_file = os.path.join(self._path, 'label.idl')
        self._classes = ('__background__',  # always index 0
                         'vehicle', 'pedestrian', 'cyclist', 'traffic_lights')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb

        self._salt = str(uuid.uuid4())
        """
        cleanup：是否清除result文件
        use_salt: 是否给result文件加salt 
        """
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'use_07_metric': False
                       }

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

    def _get_sz_results_file_template(self):
        # filename of result
        if self.config['use_salt']:
            salt = self._salt
        else:
            salt = ''
        filename = 'sz_result_to_evaluation' + salt + self._image_set + '_{:s}.txt'
        path = os.path.join(
            self._path,
            'results',
            filename)
        dirpath = os.path.join(
            self._path,
            'results')
        if os.path.exists(dirpath) is False:
            os.makedirs(dirpath)
        return path

    def _write_sz_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} sz results file'.format(cls)
            filename = self._get_sz_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_sz_results_file(all_boxes)
        result = self._do_python_eval(output_dir)
        if self.config['cleanup']:
            logging.info('cleanup the result file........')
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_sz_results_file_template().format(cls)
                os.remove(filename)
        return result

    def _do_python_eval(self, output_dir='output'):
        anno_filename = self._label_file
        image_set_file = os.path.join(self._path, self._image_set + '.txt')
        aps = []
        use_07_metric = self.config['use_07_metric']
        print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        print('~~~~~~~~')
        weighted_ap = 0
        weighted_n = 0
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_sz_results_file_template().format(cls)
            rec, prec, ap, ground_truth_box_num = sz_eval(
                filename, anno_filename, image_set_file, cls, ovthresh=0.5,
                use_07_metric=use_07_metric)
            aps += [ap]
            weighted_n += ground_truth_box_num
            weighted_ap += ground_truth_box_num * ap
            print('AP for %s = %f   (%d samples)' % (cls, ap, ground_truth_box_num))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('Weighted AP = {:.4f}'.format(float(weighted_ap) / weighted_n))
        print('~~~~~~~~')
        return np.mean(aps)
        # print('Results:')
        # for ap in aps:
        #     print('{:.3f}'.format(ap))
        # print('{:.3f}'.format(np.mean(aps)))

    def _do_mscnn_eval(self, file_dict):
        """
        file_dict: 'cls' -> 'mscnn eval file'
        """
        anno_filename = self._label_file
        image_set_file = os.path.join(self._path, self._image_set + '.txt')
        aps = []
        use_07_metric = self.config['use_07_metric']
        print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')
        print('~~~~~~~~')
        weighted_ap = 0
        weighted_n = 0
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            if cls in file_dict:
                filename = file_dict[cls]
            else:
                print('No file found for {}'.format(cls))
                continue
            rec, prec, ap, ground_truth_box_num = sz_eval(
                filename, anno_filename, image_set_file, cls, ovthresh=0.5,
                use_07_metric=use_07_metric)
            aps += [ap]
            weighted_n += ground_truth_box_num
            weighted_ap += ground_truth_box_num * ap
            print('AP for %s = %f   (%d samples)' % (cls, ap, ground_truth_box_num))
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('Weighted AP = {:.4f}'.format(float(weighted_ap) / weighted_n))
        print('~~~~~~~~')
        return np.mean(aps)


if __name__ == '__main__':
    from datasets.sz import sz
    d = sz('train')
    res = d.roidb
    print(len(res))
    from IPython import embed
    embed()
