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
    parser.add_argument('-d', '--sz-dir', default='data/ksz')
    parser.add_argument('-t', '--train-split', help="0~1", type=float, default=0.8)
    parser.add_argument('-p', '--prefix-name', help="prefix", default="")
    args = parser.parse_args()
    return args


def main(args):
    all_pics = glob(os.path.join(args.sz_dir, "training/training/*.png"))
    all_index = []
    label_file = os.path.join(args.sz_dir, 'label.idl')
    new_label_file = os.path.join(args.sz_dir, 'label_veh.idl')

    with open(label_file) as f:
        with open(new_label_file, 'w') as save_f:
            for line in f:
                j = json.loads(line)
                key = None
                for x in j:
                    key = x
                    break
                index = key.split('.')[0]
                write = False
                write_json = None
                write_json = {key: []}
                for bbox in j[key]:
                    """我们只要特定的类别"""
                    if bbox[4] == 1:
                        write = True
                        """如果修改类别号，在这里修改即可"""
                        bbox[4] = 1
                        write_json[key].append(bbox)
                if write:
                    # print('----------')
                    # print(json.dumps(write_json))
                    # print('---------')
                    save_f.write(json.dumps(write_json))
                    save_f.write('\n')
                    all_index.append(index)

    random.shuffle(all_index)
    train_number = int(len(all_index) * args.train_split)
    train_index = all_index[0:train_number]
    val_index = all_index[train_number:len(all_index)]
    with open(os.path.join(args.sz_dir, args.prefix_name + 'train_veh.txt'), 'w') as f:
        for index in train_index:
            f.write(index)
            f.write('\n')
    with open(os.path.join(args.sz_dir, args.prefix_name + 'val_veh.txt'), 'w') as f:
        for index in val_index:
            f.write(index)
            f.write('\n')



if __name__ == '__main__':
    args = parse_args()
    main(args)
    print('done.')
