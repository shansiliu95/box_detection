python detect.py --weights ./runs/train/exp/weights/best.pt --source ./304_reversed.MP4 --x-min 120 --x-max 500 --y-min 200 --y-max 384 --in-target-threshold 0.4 --same-cart-threshold 0.8 --conf-thres 0.5 --in-x -1 --in-y 1 --window-len 40

#python detect.py --weights ./runs/train/exp/weights/best.pt --source ./1238.MP4 --x-min 60 --x-max 400 --y-min 200 --y-max 384 --in-target-threshold 0.2 --same-cart-threshold 0.8 --conf-thres 0.5 --in-x -1 --in-y 1 --window-len 40
