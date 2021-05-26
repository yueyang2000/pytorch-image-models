exp_name='0525_trial'

GPU='RTX2080Ti'

gpus=8

srun -J da -N 1 -p $GPU --gres gpu:$gpus python -m torch.distributed.launch --nproc_per_node=$gpus train.py imagenet -b 64 --model resnet50 --sched cosine --epochs 200 --lr 0.2 --amp --remode pixel --reprob 0.6 --aug-splits 3 --resplit --split-bn --jsd --dist-bn reduce --workers 8 --experiment $exp_name
