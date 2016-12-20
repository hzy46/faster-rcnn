import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--veh', default='output/faster_rcnn_end2end/sz_veh_test/sz_result.json')
    parser.add_argument('-p', '--ped', default='output/faster_rcnn_end2end/sz_ped_test/sz_result.json')
    parser.add_argument('-c', '--cyc', default='output/faster_rcnn_end2end/sz_cyc_test/sz_result.json')
    parser.add_argument('-l', '--lights', default='output/faster_rcnn_end2end/sz_lights_test/sz_result.json')
    parser.add_argument('-o', '--output', default='output/faster_rcnn_end2end/output.json')
    args = parser.parse_args()
    return args


def main(args):
    veh = None
    ped = None
    cyc = None
    lights = None
    if args.veh != 'none':
        with open(args.veh) as f:
            veh = json.load(f)
    if args.ped != 'none':
        with open(args.ped) as f:
            ped = json.load(f)
    if args.cyc != 'none':
        with open(args.cyc) as f:
            cyc = json.load(f)
    if args.lights != 'none':
        with open(args.lights) as f:
            lights = json.load(f)

    if ped:
        print(args.ped)
        for key in ped:
            for box in ped[key]:
                box[4] = 2
            if key in veh:
                veh[key].extend(ped[key])
            else:
                veh[key] = ped[key]

    if cyc:
        print(args.cyc)
        for key in cyc:
            for box in cyc[key]:
                box[4] = 3
            if key in veh:
                veh[key].extend(cyc[key])
            else:
                veh[key] = cyc[key]

    if lights:
        print(args.lights)
        for key in lights:
            for box in lights[key]:
                box[4] = 20
            if key in veh:
                veh[key].extend(lights[key])
            else:
                veh[key] = lights[key]

    with open(args.output, 'w') as f:
        json.dump(veh, f)

if __name__ == '__main__':
    args = parse_args()
    main(args)
    print('done.')
