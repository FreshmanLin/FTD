# CUDA_VISIBLE_DEVICES=6 python train.py --prune_ratio=0.25 --dataset=CIFAR100
    # --learning_rate=0.00848750211771974 \
    # --batch_size=64 \
    # --momentum=0.886721185393655 \
    # --weight_decay=0.00194231748520205 \
    # --mix_ratio=0.797158827707392 \
CUDA_VISIBLE_DEVICES=0 python train.py --prune_ratio=0.25 --dataset=CIFAR10
wait
CUDA_VISIBLE_DEVICES=0 python train.py --prune_ratio=0.5 --dataset=CIFAR10
wait
CUDA_VISIBLE_DEVICES=0 python train.py --prune_ratio=0.75 --dataset=CIFAR10