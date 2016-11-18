# export CHECKPOINT_PATH=XXXX
./tools/test_net.py --gpu 0  --cfg experiments/cfgs/faster_rcnn_end2end.yml --network VGGnet_test --imdb sz_val --weights "${CHECKPOINT_PATH}"