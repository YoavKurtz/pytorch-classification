python imagenet.py -a l_resnet110 --data $1 --epochs 300 --schedule 150 225 --gamma 0.1 \
 --train-batch 128 --test-batch 128 --wd 5e-4 --reg-type $2 --gpu-id $3 $4