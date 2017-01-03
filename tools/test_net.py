#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""

import _init_paths
from fast_rcnn.test import test_net
from fast_rcnn.config import cfg, cfg_from_file
from datasets.factory import get_imdb
from networks.factory import get_network
import argparse
import pprint
import time
import os
import sys
import tensorflow as tf
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
    parser.add_argument('--weights', dest='model',
                        help='model to test',
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

    while not os.path.exists(args.model) and args.wait:
        print('Waiting for {} to exist...'.format(args.model))
        time.sleep(10)

    weights_filename = os.path.splitext(os.path.basename(args.model))[0]

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

    network = get_network(args.network_name, n_classes)
    print 'Use network `{:s}` in training'.format(args.network_name)

    cfg.GPU_ID = args.gpu_id

    with tf.variable_scope('custom', reuse=False):
        bbox_means = tf.get_variable("bbox_means", shape=(n_classes * 4, ), trainable=False)
        bbox_stds = tf.get_variable("bbox_stds", shape=(n_classes * 4, ), trainable=False)

    saver = tf.train.Saver()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    saver.restore(sess, args.model)
    print ('Loading model weights from {:s}').format(args.model)

    test_net(sess, network, imdb, weights_filename, vis=args.vis)
