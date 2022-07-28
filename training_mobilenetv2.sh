mkdir -p output
python3 train.py -trd /data/cifar10c/train -ted /data/cifar10c/test -e 1 -m mobilenetv2
