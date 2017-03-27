# coding: utf-8
import argparse
import json
from glob import glob
import os
import subprocess
import time
import traceback
import re
import copy

def glob_all(dir_path):
    ckpt_list = []
    ckpt_list.extend(glob(os.path.join(dir_path, '*.ckpt')))
    inside = os.listdir(dir_path)
    for dir_name in inside:
        if os.path.isdir(os.path.join(dir_path, dir_name)):
            ckpt_list.extend(glob_all(os.path.join(dir_path, dir_name)))
    return ckpt_list


def glob_all_dict(dir_path):
    ckpt_list = glob_all(dir_path)
    ckpt_dict = {}
    for ckpt in ckpt_list:
        ckpt_dict[ckpt] = True
    return ckpt_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--ckpt-dir', default='output/faster_rcnn_end2end/')
    """加入事先执行的，这个是关键词搜索，有这个关键词的就去执行"""
    parser.add_argument('-e', '--eval-old', default=None, help='example: sz_lights_train,sz_cyc_train')
    parser.add_argument('-w', '--wait', default="TRUE", help="whether to wait for new ckpt after finishing eval-old")
    args = parser.parse_args()
    return args


def add_report(ckpt):
    """
    example:
    ckpt: output/faster_rcnn_end2end/sz_cyc_train/VGGnet_fast_rcnn_iter_5000.ckpt
    """
    iter_number = int(ckpt.split('.')[-2].split('_')[-1])
    model_name = ckpt.split('/')[-2]
    model_class = model_name.replace('sz_', '').replace('_trainval', '').replace('_train', '')
    save_file_name = model_name + '.txt'
    save_file_name = os.path.join(args.ckpt_dir, save_file_name)
    if os.path.exists(save_file_name) is False:
        with open(save_file_name, 'w'):
            pass
    """replace"""
    imdb_val = model_name.replace('trainval', 'valall').replace('train', 'valall')
    command = './tools/test_net.py --gpu 0  --cfg experiments/cfgs/faster_rcnn_end2end_%s.yml '\
        '--network VGGnet_test --imdb %s --weights %s' % (model_class, imdb_val, ckpt)
    print(iter_number, model_name, command)
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    out, err = p.communicate()
    result = out.splitlines()
    AP = float(result[-2].decode('utf-8').split(' ')[-1])
    with open(save_file_name, 'a') as f:
        f.write('%s %s iter: %d AP: %f\n' % (time.strftime('%Y-%m-%d %H:%M'), model_name, iter_number, AP))



def ckpt_cmp(ckpt1, ckpt2):
    iter_number1 = int(ckpt1.split('.')[-2].split('_')[-1])
    iter_number2 = int(ckpt2.split('.')[-2].split('_')[-1])
    if iter_number1 < iter_number2:
        return 1
    elif iter_number1 > iter_number2:
        return -1
    else:
        return 0

def get_list_to_eval(eval_old, ckpt_dict):
    """
    eval_old: lights:1000,2000,3000;veh:20000,30000
                   also support regex: lights:.*;veh:20000,30000
                   因为有分号，防止linux识别为多条命令，必须加引号！！
    ckpt_dict: {"path_to_ckpt": True}
    """
    to_eval = []
    ckpt_dict_temp = copy.copy(ckpt_dict)
    eval_old = eval_old.split(';')
    for item in eval_old:
        item = item.strip()
        if item == "":
            continue
        main_keyword = item.split(':')[0].strip()
        main_keyword_pattern = re.compile(main_keyword)
        sub_keyword_list = item.split(':')[1].strip().split(',')
        for one_word in sub_keyword_list:
            one_word = one_word.strip()
            one_word_pattern = re.compile(one_word)
            for ckpt in ckpt_dict_temp:
                if ckpt_dict_temp[ckpt] is True and main_keyword_pattern.search(ckpt) and one_word_pattern.search(ckpt):
                    to_eval.append(ckpt)
                    ckpt_dict_temp[ckpt] = False
    return to_eval


def main(args):
    ckpt_dict = glob_all_dict(args.ckpt_dir)
    if args.eval_old:
        do_list = get_list_to_eval(args.eval_old, ckpt_dict)
        do_list.sort(cmp=ckpt_cmp, reverse=True)
        for ckpt in do_list:
            print("Will evaluate: %s" % ckpt)
        for ckpt in do_list:
            try:
                add_report(ckpt)
            except Exception:
                traceback.print_exc()
    if args.wait.upper().startswith('TRUE'):
        args.wait = True
    else:
        args.wait = False
    if args.wait is False:
        print("Done.")
        return
    print('Now watching.....')
    while True:
        time.sleep(30)
        new_ckpt_dict = glob_all_dict(args.ckpt_dir)
        do_list = []
        for ckpt in new_ckpt_dict:
            if ckpt not in ckpt_dict:
                do_list.append(ckpt)
        do_list.sort(cmp=ckpt_cmp, reverse=True)
        for ckpt in do_list:
            try:
                add_report(ckpt)
                time.sleep(5)
            except Exception:
                traceback.print_exc()
        ckpt_dict = new_ckpt_dict
    # add_report('output/faster_rcnn_end2end/sz_cyc_train/VGGnet_fast_rcnn_iter_5000.ckpt')

if __name__ == '__main__':
    args = parse_args()
    main(args)
