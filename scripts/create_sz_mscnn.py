# coding:utf-8
from __future__ import absolute_import
import argparse
import os
import traceback
import random
from glob import glob
import json
import shutil

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--target-dir')
    parser.add_argument('-t', '--train-split', help="0~1", type=float, default=0.8)
    parser.add_argument('-r', '--remove-empty', default=True, type=bool)
    args = parser.parse_args()
    return args


def main(args):
    input()
    val_dir = os.path.join(args.sz_dir, 'val')
    train_dir = os.path.join(args.sz_dir, 'train')
    if os.path.exists(os.path.join(val_dir)) is False:
        os.makedirs(os.path.join(train_dir))

    all_pics = glob(os.path.join(args.sz_dir, "training/training/*.jpg"))
    all_index = [os.path.basename(filepath).split('.')[0] for filepath in all_pics]
    random.shuffle(all_index)
    train_number = int(len(all_index) * args.train_split)
    train_index = all_index[0:train_number]
    val_index = all_index[train_number:len(all_index)]
    label_file = os.path.join(args.sz_dir, 'label.idl')
    with open(label_file) as f:
        image_dict = {}  # index -> number
        for line in f:
            j = json.loads(line)
            key = j.keys()[0]
            image_dict[key.split('.')[0]] = len(j[key])
        if args.remove_empty is True:
            print('---------------remove images without bbox in TRAINING dataset----------------')
        with open(os.path.join(args.sz_dir, args.prefix_name + 'train.txt'), 'w') as f:
            for index in train_index:
                if args.remove_empty is True and image_dict[index] == 0:
                    continue
                origin_file = os.path.join(args.sz_dir, all_index + '.jpg')
                shutil.copy(origin_file, os.path.join(train_dir, all_index + '.jpg'))
        with open(os.path.join(args.sz_dir, args.prefix_name + 'val.txt'), 'w') as f:
            for index in val_index:
                origin_file = os.path.join(args.sz_dir, all_index + '.jpg')
                shutil.copy(origin_file, os.path.join(train_dir, all_index + '.jpg'))

if __name__ == '__main__':
    args = parse_args()
    main(args)
    print('done.')
