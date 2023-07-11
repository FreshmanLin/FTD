import argparse
import torch
import os
import sys; sys.path.append("..")
from torchvision import datasets, transforms
from utils.initialize import initialize
from utils.utils_baseline import get_dataset, get_network, get_eval_pool, epoch, get_time, DiffAugment, augment, ParamDiffAug, TensorDataset

from dataset_syn import distilled_dataset
import tqdm
import kornia as K
import copy
import time
import torch.nn as nn
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES']='2'

def epoch_mix(mode, dataloader1, dataloader2, net, optimizer, criterion, args, aug, texture=False):
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.to(args.device)

    # if args.dataset == "ImageNet":
    #     class_map = {x: i for i, x in enumerate(config.img_net_classes)}

    if mode == 'train':
        net.train()
    else:
        net.eval()

    dataloader_iterator = iter(dataloader2)

    for i_batch, datum1 in enumerate(dataloader1):
        img1 = datum1[0].float().to(args.device)
        lab1 = datum1[1].long().to(args.device)

        try:
            datum2 = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(dataloader2)
            datum2 = next(dataloader_iterator)

        img2 = datum2[0].float().to(args.device)
        lab2 = datum2[1].long().to(args.device)

        img = torch.cat([img1,img2], dim=0)
        lab = torch.cat([lab1,lab2], dim=0)

        shuffle_index = torch.randperm(img.size(0))
        img = img[shuffle_index]
        lab = lab[shuffle_index]

        if mode == "train" and texture:
            img = torch.cat([torch.stack([torch.roll(im, (torch.randint(args.im_size[0]*args.canvas_size, (1,)), torch.randint(args.im_size[0]*args.canvas_size, (1,))), (1,2))[:,:args.im_size[0],:args.im_size[1]] for im in img]) for _ in range(args.canvas_samples)])
            lab = torch.cat([lab for _ in range(args.canvas_samples)])

        if aug:
            if args.dsa:
                img = DiffAugment(img, args.dsa_strategy, param=args.dsa_param)
            else:
                img = augment(img, args.dc_aug_param, device=args.device)

        # if args.dataset == "ImageNet" and mode != "train":
        #     lab = torch.tensor([class_map[x.item()] for x in lab]).to(args.device)

        n_b = lab.shape[0]

        output = net(img)
        loss = criterion(output, lab)

        acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

        loss_avg += loss.item()*n_b
        acc_avg += acc
        num_exp += n_b

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg


def evaluate_synset(it_eval, net, images_train, images_train_syn, labels_train, labels_train_syn, testloader, args, return_loss=False, texture=False):
    net = net.to(args.device)
    images_train = images_train.to(args.device)
    images_train_syn = images_train_syn.to(args.device)
    labels_train = labels_train.to(args.device)
    labels_train_syn = labels_train_syn.to(args.device)

    lr = float(args.lr_net)
    Epoch = int(args.epoch_eval_train)
    lr_schedule = [Epoch//2+1]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    criterion = nn.CrossEntropyLoss().to(args.device)

    dst_train = TensorDataset(images_train, labels_train)
    dst_train_syn = TensorDataset(images_train_syn, labels_train_syn)

    batch_size = int(0.9 * args.batch_train)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size = batch_size, shuffle=True, num_workers=0)
    trainloader_syn = torch.utils.data.DataLoader(dst_train_syn, batch_size = args.batch_train-batch_size, shuffle=True, num_workers=0)

    start = time.time()
    acc_train_list = []
    loss_train_list = []

    for ep in tqdm.tqdm(range(Epoch+1)):
        loss_train, acc_train = epoch_mix('train', trainloader, trainloader_syn, net, optimizer, criterion, args, aug=True, texture=texture)
        acc_train_list.append(acc_train)
        loss_train_list.append(loss_train)
        if ep == Epoch:
            with torch.no_grad():
                loss_test, acc_test = epoch('test', testloader, net, optimizer, criterion, args, aug=False)
        if ep in lr_schedule:
            lr *= 0.1
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)


    time_train = time.time() - start

    print('%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f' % (get_time(), it_eval, Epoch, int(time_train), loss_train, acc_train, acc_test))

    if return_loss:
        return net, acc_train_list, acc_test, loss_train_list, loss_test
    else:
        return net, acc_train_list, acc_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')

    parser.add_argument('--subset', type=str, default='imagenette', help='ImageNet subset. This only does anything when --dataset=ImageNet')

    parser.add_argument('--model', type=str, default='ConvNet', help='model')

    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')

    parser.add_argument('--eval_mode', type=str, default='S',
                        help='eval_mode, check utils.py for more info')

    parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')

    parser.add_argument('--eval_it', type=int, default=100, help='how often to evaluate')

    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=5000, help='how many distillation steps to perform')

    parser.add_argument('--lr_img', type=float, default=1000, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_lr', type=float, default=1e-05, help='learning rate for updating... learning rate')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='initialization for synthetic learning rate')

    parser.add_argument('--lr_init', type=float, default=0.01, help='how to init lr (alpha)')

    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_syn', type=int, default=None, help='should only use this if you run out of VRAM')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')

    parser.add_argument('--pix_init', type=str, default='real', choices=["noise", "real"],
                        help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')

    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')

    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')

    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')

    parser.add_argument('--expert_epochs', type=int, default=3, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=20, help='how many steps to take on synthetic data')
    parser.add_argument('--max_start_epoch', type=int, default=25, help='max epoch we can start at')

    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")

    parser.add_argument('--load_all', action='store_true', help="only use if you can fit all expert trajectories into RAM")

    parser.add_argument('--no_aug', type=bool, default=False, help='this turns off diff aug during distillation')

    parser.add_argument('--texture', action='store_true', help="will distill textures instead")
    parser.add_argument('--canvas_size', type=int, default=2, help='size of synthetic canvas')
    parser.add_argument('--canvas_samples', type=int, default=1, help='number of canvas samples per iteration')


    parser.add_argument('--max_files', type=int, default=None, help='number of expert files to read (leave as None unless doing ablations)')
    parser.add_argument('--max_experts', type=int, default=None, help='number of experts to read per file (leave as None unless doing ablations)')

    parser.add_argument('--force_save', action='store_true', help='this will save images for 50ipc')
    parser.add_argument('--ema_decay', type=float, default=0.999)

    args = parser.parse_args()

    # initialize(args, seed=42)
    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform = transforms.Compose([transforms.ToTensor()])
    dst_train_syn = distilled_dataset(transform)
    dataset = 'CIFAR100'
    # dataset = Cifar_syn(args.batch_size, args.threads)
    # log = Log(log_each=10)
    model_eval = 'ConvNet'
    if dataset == 'CIFAR100':
        channel = 3
        num_classes = 100
        im_size=(32,32)
        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)
    args.im_size = im_size[0]

    if args.dsa:
        # args.epoch_eval_train = 1000
        args.dc_aug_param = None

    args.dsa_param = ParamDiffAug()

    dsa_params = args.dsa_param
    dst_train = datasets.CIFAR100('data', train=True, download=True, transform=transform)
    dst_test = datasets.CIFAR100('data', train=False, download=True, transform=transform)

    # zca
    images = []
    labels = []
    print("Train ZCA")
    for i in tqdm.tqdm(range(len(dst_train))):
        im, lab = dst_train[i]
        images.append(im)
        labels.append(lab)
    images = torch.stack(images, dim=0).to(args.device)
    labels = torch.tensor(labels, dtype=torch.long, device="cpu")
    zca = K.enhance.ZCAWhitening(eps=0.1, compute_inv=True)
    zca.fit(images)
    zca_images = zca(images).to("cpu")
    dst_train = TensorDataset(zca_images, labels)

    images = []
    labels = []
    print("Test ZCA")
    for i in tqdm.tqdm(range(len(dst_test))):
        im, lab = dst_test[i]
        images.append(im)
        labels.append(lab)
    images = torch.stack(images, dim=0).to(args.device)
    labels = torch.tensor(labels, dtype=torch.long, device="cpu")

    zca_images = zca(images).to("cpu")
    dst_test = TensorDataset(zca_images, labels)

    testloader = torch.utils.data.DataLoader(dst_test, batch_size=128, shuffle=False, num_workers=2)

    image_syn_eval = copy.deepcopy(dst_train_syn.data.detach())
    label_syn_eval = copy.deepcopy(dst_train_syn.labels.detach())

    image_origin_eval = copy.deepcopy(dst_train.images)
    label_origin_eval = copy.deepcopy(dst_train.labels)

    args.lr_net = torch.tensor(0.06).to(args.device)
    

    _, acc_eval, acc_eavl = evaluate_synset(0, net_eval, 
                                            images_train = image_origin_eval, 
                                            images_train_syn = image_syn_eval, 
                                            labels_train = label_origin_eval,
                                            labels_train_syn = label_syn_eval, 
                                            testloader = testloader, 
                                            args = args, texture=args.texture)

   
