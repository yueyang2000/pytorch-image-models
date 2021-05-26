exp_name='0526_moex'

GPU='RTX2080Ti'

gpus=8

norm_method=pono
lam=0.9
prob=0.25
arch=moex_resnet50

srun -J da -N 1 -p $GPU --gres gpu:$gpus python -m torch.distributed.launch --nproc_per_node=$gpus train.py imagenet -b 64 --model $arch --sched cosine --epochs 200 --lr 0.2 --amp --remode pixel --reprob 0.6 --resplit --split-bn  --dist-bn reduce --workers 8 --cutmix 1.0 --moex_norm ${norm_method} --moex_lam ${lam} --moex_prob ${prob} --experiment $exp_name
