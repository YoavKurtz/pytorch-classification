python imagenet.py -a l_resnet110 --data cifar100 --epochs 300 --schedule 150 225 --gamma 0.1 -c checkpoints/cifar100/resnet110 --train-batch 128 --test-batch 128 --wd 5e-4