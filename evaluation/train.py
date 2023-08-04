import sys; sys.path.append("..")
import argparse
import torch

import os
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset
import torch.nn.functional as F
# from model.smooth_cross_entropy import smooth_crossentropy
# from model.networks import ConvNet3
# from data.cifar import Cifar100
# from data.cifar_syn import Cifar_syn
from model.resnet import ResNet18
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from utility.bypass_bn import enable_running_stats, disable_running_stats
from utility.cutout import Cutout
import numpy as np

from utils.utils_baseline import get_dataset, get_network, get_eval_pool, epoch, get_time, DiffAugment, augment, ParamDiffAug, TensorDataset
from dataset_syn import distilled_dataset


# from sam import SAM
# os.environ['CUDA_VISIBLE_DEVICES']='4'

def _get_statistics(cifar):
    if cifar=="CIFAR100":
        train_set = torchvision.datasets.CIFAR100(root='data', train=True, download=False, transform=transforms.ToTensor())
    else:
        train_set = torchvision.datasets.CIFAR10(root='data', train=True, download=False, transform=transforms.ToTensor())

    data = torch.cat([d[0] for d in DataLoader(train_set)])
    return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.05, type=float, help="Base learning rate at the start of the training.")
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
    parser.add_argument("--prune_ratio", default=0,type=float, help="The ratio of dataset preserved in the following experiment.")
    parser.add_argument("--dataset", default="CIFAR100",type=str, help="The dataset is CIFAR10 or CIFAR100.")
    args = parser.parse_args()

    print('Hyper-parameters: \n', args.__dict__)

    def smooth_crossentropy(pred, gold, smoothing=0.1):
        n_class = pred.size(1)

        one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
        one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
        log_prob = F.log_softmax(pred, dim=1)

        return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1)
    initialize(args, seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("torch.cuda.is_available() =",torch.cuda.is_available())
    print("torch.cuda.device_count() =",torch.cuda.device_count())
    # dataset = Cifar_syn(args.batch_size, args.threads)

    # transform = transforms.Compose([transforms.ToTensor()])
    cifar = args.dataset
    dst_train_syn = distilled_dataset(cifar)
    # mean = [0.5071, 0.4866, 0.4409]
    # std = [0.2673, 0.2564, 0.2762]
    mean, std = _get_statistics(cifar)
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
    if cifar=="CIFAR100":
        dst_train = datasets.CIFAR100('data', train=True, download=True, transform=train_transform)
        dst_test = datasets.CIFAR100('data', train=False, download=True, transform=test_transform)
        classes=100
    else:
        dst_train = datasets.CIFAR10('data', train=True, download=True, transform=train_transform)
        dst_test = datasets.CIFAR10('data', train=False, download=True, transform=test_transform)        
        classes=10
    r = args.prune_ratio
    print("Prune ratio =", r)
    if r>0:
        indices = np.load(f"sorted_dataset_indexes_{cifar}.npy", allow_pickle=True)
        r_indices = indices[:int((1-r)*len(indices))]
        dst_train = Subset(dst_train, r_indices)
    

    if args.mix_batch:
        batch_size = int(args.mix_ratio * args.batch_size)
        train_loader = DataLoader(dst_train, batch_size=batch_size, shuffle=True, num_workers=args.threads)
        train_loader_syn = DataLoader(dst_train_syn, batch_size=args.batch_size-batch_size, shuffle=True, num_workers=args.threads)
    else:
        dst_mix = ConcatDataset([dst_train, dst_train_syn])
        train_loader = torch.utils.data.DataLoader(dst_mix, batch_size = args.batch_train, shuffle=True, num_workers=args.threads)
    test_loader = DataLoader(dst_test, batch_size=args.batch_size, shuffle=False, num_workers=args.threads)


    log = Log(log_each=10)
    # model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10).to(device)
    model = ResNet18(classes).to(device)
    # model = ConvNet3(3, 100).to(device)
    # model_eval = 'ResNet18'
    # channel = 3
    # num_classes = 100
    # im_size=(32,32)
    # model = get_network(model_eval, channel, num_classes, im_size).to(device)
    base_optimizer = torch.optim.SGD
    # optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    # scheduler = StepLR(optimizer, args.learning_rate, args.epochs)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_acc = 0
    best_iter = 0
    for epoch in range(args.epochs):
        model.train()
        log.train(len_dataset=len(train_loader))
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
                log(model, loss.cpu(), correct.cpu(), scheduler.get_last_lr()[0])
                scheduler.step(epoch)

        model.eval()
        log.eval(len_dataset=len(test_loader))
        correct_num = 0
        total_num = 0
        with torch.no_grad():
            for batch in test_loader:
                inputs, targets = (b.to(device) for b in batch)

                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                log(model, loss.cpu(), correct.cpu())
                correct_num+=correct.cpu().sum().item()
                total_num+=inputs.size(0)
        acc = correct_num/total_num
        if acc > best_acc:
            best_acc = acc
            best_iter = epoch
    log.flush()

    print("Best test accuracy is:", best_acc)
    print("Best result is from:", best_iter)