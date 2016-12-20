#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""

import _init_paths
from datasets.factory import get_imdb
import argparse
import logging


file_dict = {
    'vehicle': '../mscnn/examples/kitti_car/detections/result_kitti_8s_768_35k_test_4classes_val.txt'
}


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='sz_val', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    imdb = get_imdb(args.imdb_name)
    imdb._do_mscnn_eval(file_dict)
