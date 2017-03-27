#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""

import _init_paths
from fast_rcnn.test_ensemble import test_net
from fast_rcnn.config import cfg, cfg_from_file, get_output_dir
from datasets.factory import get_imdb
import argparse
import pprint
import time
import os
import sys
import logging


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default='vgg_test', type=str)
    parser.add_argument('--vis', type=bool, default=False)
    parser.add_argument('--raw-rcnn-data', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Using config:')
    pprint.pprint(cfg)

    imdb = get_imdb(args.imdb_name)

    device_name = '/gpu:{:d}'.format(args.gpu_id)
    print device_name

    if args.imdb_name.startswith('sz_veh') or args.imdb_name.startswith('sz_cyc') or args.imdb_name.startswith('sz_ped') or args.imdb_name.startswith('sz_lights'):
        n_classes = 2
    elif args.imdb_name.startswith('ksz_veh') or args.imdb_name.startswith('ksz_cyc') or args.imdb_name.startswith('ksz_ped') or args.imdb_name.startswith('ksz_lights'):
        n_classes = 2
    elif args.imdb_name.startswith('sz'):
        n_classes = 5
    elif args.imdb_name.startswith('voc'):
        n_classes = 21
    else:
        raise Exception('Give me the correct n_classes of %s' % (args.imdb_name))

    assert len(imdb._classes) == n_classes

    cfg.GPU_ID = args.gpu_id

    rcnn_raw_data_file_list = args.raw_rcnn_data.split(',')
    print('Use rcnn_raw_data_file: %s' % rcnn_raw_data_file_list)
    test_net(rcnn_raw_data_file_list, imdb, vis=args.vis)
