# coding:utf-8
from __future__ import absolute_import
import argparse
import os
from glob import glob


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--sz-dir', default='data/sz')
    args = parser.parse_args()
    return args


def main(args):
    all_pics = glob(os.path.join(args.sz_dir, "testing/testing/*.jpg"))
    all_index = [os.path.basename(filepath).split('.')[0] for filepath in all_pics]
    with open(os.path.join(args.sz_dir, 'test.txt'), 'w') as f:
        for index in all_index:
            f.write(index)
            f.write('\n')

if __name__ == '__main__':
    args = parse_args()
    main(args)
    print('done.')
