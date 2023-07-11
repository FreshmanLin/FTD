import argparse
import torch
import os
import sys; sys.path.append("..")
from torchvision import datasets, transforms
from utils.initialize import initialize
from utils.utils_baseline import get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, ParamDiffAug, TensorDataset

from dataset_syn import distilled_dataset
import tqdm
import kornia as K
import copy

os.environ['CUDA_VISIBLE_DEVICES']='2'
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

    args.lr_net = torch.tensor(0.06).to(args.device)
    
    _, acc_eval, acc_eavl = evaluate_synset(0, net_eval, image_syn_eval, label_syn_eval, testloader, args, texture=args.texture)

   
