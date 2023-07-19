import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

import sys; sys.path.append("..")
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats
from utility.cutout import Cutout

from utils.utils_baseline import get_dataset, get_network, get_eval_pool, epoch, get_time, DiffAugment, augment, ParamDiffAug, TensorDataset
from dataset_syn import distilled_dataset

from model.resnet import ResNet18
os.environ['CUDA_VISIBLE_DEVICES']='1'

def smooth_crossentropy(pred, gold, smoothing=0.1):
    n_class = pred.size(1)

    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
    log_prob = F.log_softmax(pred, dim=1)

    return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1)

def train_mix(args):
    args = argparse.Namespace(**args)
    initialize(args, seed=42)
    print("torch.cuda.is_available() =",torch.cuda.is_available())
    print("torch.cuda.device_count() =",torch.cuda.device_count())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset = Cifar_syn(args.batch_size, args.threads)

    # transform = transforms.Compose([transforms.ToTensor()])
    dst_train_syn = distilled_dataset()
    mean = [0.5071, 0.4866, 0.4409]
    std = [0.2673, 0.2564, 0.2762]
    train_transform = transforms.Compose([
        torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        Cutout()
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    dst_train = datasets.CIFAR100('data', train=True, download=True, transform=train_transform)
    dst_test = datasets.CIFAR100('data', train=False, download=True, transform=test_transform)

    if args.mix_batch:
        batch_size = int(args.mix_ratio * args.batch_size)
        train_loader = DataLoader(dst_train, batch_size=batch_size, shuffle=False, num_workers=args.threads)
        train_loader_syn = DataLoader(dst_train_syn, batch_size=args.batch_size-batch_size, shuffle=False, num_workers=args.threads)
    else:
        dst_mix = ConcatDataset([dst_train, dst_train_syn])
        train_loader = torch.utils.data.DataLoader(dst_mix, batch_size = args.batch_train, shuffle=True, num_workers=args.threads)
    test_loader = DataLoader(dst_test, batch_size=args.batch_size, shuffle=False, num_workers=args.threads)


    # log = Log(log_each=10)
    # model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10).to(device)
    model = ResNet18(100).to(device)
    # model = ConvNet3(3, 100).to(device)
    # model_eval = 'ConvNet'
    # channel = 3
    # num_classes = 100
    # im_size=(32,32)
    # model = get_network(model_eval, channel, num_classes, im_size).to(device)
    base_optimizer = torch.optim.SGD
    # optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, args.learning_rate, args.epochs)

    for epoch in range(args.epochs):
        model.train()
        # log.train(len_dataset=len(train_loader))
        if args.mix_batch:
            dataloader_iterator = iter(train_loader_syn)
        for batch in train_loader:
            inputs, targets = (b.to(device) for b in batch)
            if args.mix_batch:
                try:
                    batch2 = next(dataloader_iterator)
                except StopIteration:
                    dataloader_iterator = iter(train_loader_syn)
                    batch2 = next(dataloader_iterator)
                inputs2, targets2 = (b.to(device) for b in batch2)

                inputs = torch.cat([inputs, inputs2], dim = 0)
                targets = torch.cat([targets, targets2], dim = 0)

                shuffle_index = torch.randperm(inputs.size(0))
                inputs = inputs[shuffle_index]
                targets = targets[shuffle_index]

            optimizer.zero_grad()

            # first forward-backward step
            # enable_running_stats(model)
            predictions = model(inputs)
            # loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(predictions, targets)
            # import pdb;pdb.set_trace()
            # loss.mean().backward()
            loss.backward()
            # optimizer.first_step(zero_grad=True)
            optimizer.step()

            # second forward-backward step
            # disable_running_stats(model)
            # smooth_crossentropy(model(inputs), targets, smoothing=args.label_smoothing).mean().backward()
            # optimizer.second_step(zero_grad=True)

            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                # log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                scheduler(epoch)

        model.eval()
        # log.eval(len_dataset=len(test_loader))
        correct_num = 0
        total_num = 0
        with torch.no_grad():
            for batch in test_loader:
                inputs, targets = (b.to(device) for b in batch)

                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                # log(model, loss.cpu(), correct.cpu())
                correct_num+=correct.cpu().sum().item()
                total_num+=inputs.size(0)
        acc = correct_num/total_num
        tune.report(mean_accuracy=acc)
    # log.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--rho", default=2.0, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                    help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                    help='differentiable Siamese augmentation strategy')
    parser.add_argument('--mix_batch', type=str, default='True', choices=['True', 'False'],
                        help='Mix synthetic dataset and original dataset in each minibatch or overall.')
    parser.add_argument('--mix_ratio', type=float, default=0.9)
    args = parser.parse_args()

    # print('Hyper-parameters: \n', args.__dict__)
    config = args.__dict__
    search_space = {
        "learning_rate": tune.loguniform(1e-3, 1e-1),
        "batch_size": tune.choice([128, 256]),
        "momentum": tune.uniform(0.8, 0.99),
        "weight_decay": tune.uniform(0, 0.001),
        "mix_ratio": tune.uniform(0.8,0.95)
    }
    
    # search_space.update(args.__dict__)
    config.update(search_space)
    ray.init(num_cpus=4, num_gpus=1)

    print("Total search space:", config)
    analysis = tune.run(
        train_mix,
        num_samples=30,
        scheduler=ASHAScheduler(metric="mean_accuracy", mode="max",max_t=200),
        config=config,
        resources_per_trial={"GPU": 1, "CPU": 4},
    )

    best_trial = analysis.get_best_trial(metric="mean_accuracy", mode="max", scope="all")   # Get best trial
    print("Best trial is:", best_trial)
    best_config = analysis.get_best_config(metric="mean_accuracy", mode="max",scope="all")  # Get best trial's hyperparameters
    print("Best config is:", best_config)
    # best_logdir = analysis.best_logdir  # Get best trial's logdir
    # best_checkpoint = analysis.best_checkpoint  # Get best trial's best checkpoint
    best_result = best_trial.last_result # Get best trial's last results
    print("Best result is:", best_result)

