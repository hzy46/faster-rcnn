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
    parser.add_argument('-t', '--train-split', help="0~1", type=float, default=0.8)
    parser.add_argument('-p', '--prefix-name', help="prefix", default="")
    parser.add_argument('-r', '--remove-duplicate', default=True, type=bool)
    args = parser.parse_args()
    return args


def main(args):
    all_pics = glob(os.path.join(args.sz_dir, "training/training/*.jpg"))
    all_index = [os.path.basename(filepath).split('.')[0] for filepath in all_pics]
    random.shuffle(all_index)
    train_number = int(len(all_index) * args.train_split)
    train_index = all_index[0:train_number]
    val_index = all_index[train_number:len(all_index)]
    label_file = os.path.join(args.sz_dir, 'label.idl')
    if args.remove_duplicate is False:
        with open(os.path.join(args.sz_dir, args.prefix_name + 'train.txt'), 'w') as f:
            for index in train_index:
                f.write(index)
                f.write('\n')
    else:
        print('---------------remove images without bbox in TRAINING dataset----------------')
        with open(label_file) as f:
            image_dict = {}  # index -> number
            for line in f:
                j = json.loads(line)
                key = j.keys()[0]
                image_dict[key.split('.')[0]] = len(j[key])
        with open(os.path.join(args.sz_dir, args.prefix_name + 'train.txt'), 'w') as f:
            for index in train_index:
                if image_dict[index] != 0:
                    f.write(index)
                    f.write('\n')
    with open(os.path.join(args.sz_dir, args.prefix_name + 'val.txt'), 'w') as f:
        for index in val_index:
            f.write(index)
            f.write('\n')

if __name__ == '__main__':
    args = parse_args()
    main(args)
    print('done.')
