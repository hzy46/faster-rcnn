# coding:utf-8
from __future__ import absolute_import
import argparse
import os
import traceback
import random
from glob import glob
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--sz-dir', default='data/sz')
    args = parser.parse_args()
    return args


def main(args):
    all_pics = glob(os.path.join(args.sz_dir, "training/training/*.jpg"))
    all_index = [os.path.basename(name).split('.')[0] for name in all_pics]
    all_index_dict = {}
    for index in all_index:
        all_index_dict[index] = False
    class_ = 'lights'
    target_file = 'train_%s.txt' % class_
    save_file = 'valall_%s.txt' % class_
    with open(os.path.join(args.sz_dir, target_file)) as f:
        for index in f:
            index = index.strip()
            if index == "":
                continue
            all_index_dict[index] = True
    with open(os.path.join(args.sz_dir, save_file), 'w') as f:
        for index in all_index_dict:
            if all_index_dict[index] is False:
                f.write(index)
                f.write('\n')

if __name__ == '__main__':
    args = parse_args()
    main(args)
    print('done.')
