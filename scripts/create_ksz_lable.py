from glob import glob
import os
import json

if __name__ == '__main__':
    root_dir = '/home/dl/ssd2/dataset/kitti/'
    all_lables = glob(os.path.join(root_dir, 'label', '*.txt'))
    with open(os.path.join(root_dir, 'label.idl'), 'w') as write_f:
        for label_file in all_lables:
            with open(label_file) as f:
                j = {}
                image_name = os.path.basename(label_file.split('.')[0]) + '.png'
                j[image_name] = []
                for line in f:
                    line = line.split(' ')
                    label = line[0]
                    if label == 'Car' or label == 'Van' or label == 'Truck' or label == 'Tram':
                        label = 1
                    elif label == 'Pedestrian' or label == 'Person_sitting':
                        label = 2
                    elif label == 'Cyclist':
                        label = 3
                    elif label == 'DontCare' or label == 'Misc':
                        continue
                    else:
                        raise Exception(label)
                    bbox = [float(line[4]), float(line[5]), float(line[6]), float(line[7]), label]
                    j[image_name].append(bbox)
                write_f.write(json.dumps(j) + '\n')