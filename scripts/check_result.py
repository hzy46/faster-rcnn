import argparse
import json
import os
from glob import glob


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', default='output/faster_rcnn_end2end/output.json')
    parser.add_argument('-d', '--sz-dir', default='data/sz')
    args = parser.parse_args()
    return args


def main(args):
    test_image = glob(os.path.join(args.sz_dir, 'testing/testing/', '*.jpg'))
    print(len(test_image))
    test_image_dict = {}
    for key in test_image:
        test_image_dict[os.path.basename(key).strip()] = True
    with open(args.output) as f:
        j = json.load(f)

    count = 0
    for image_name in j:
        image_name = image_name.strip()
        count += 1
        assert image_name in test_image_dict
        for box in j[image_name]:
            assert 0 <= box[0] <= 640
            assert 0 <= box[1] <= 360
            assert 0 <= box[2] <= 640
            assert 0 <= box[3] <= 360
            assert box[4] in [1, 2, 3, 20]
            assert 0 <= box[5] <= 1
    print(count)


if __name__ == '__main__':
    args = parse_args()
    main(args)
    print('done.')
