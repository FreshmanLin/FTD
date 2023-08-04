import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import tqdm
import kornia as K
import sys; sys.path.append("..")
from utils.utils_baseline import get_dataset, get_network, get_eval_pool, epoch, get_time, DiffAugment, augment, ParamDiffAug, TensorDataset


class distilled_dataset(Dataset):
    def __init__(self, cifar):
        # mean = [0.4914, 0.4822, 0.4465]
        # std = [0.2023, 0.1994, 0.2010]
        # imgs = unnormalize(torch.load(pt_dir), mean, std)
        # self.pt_dir = "/home/stulium/FlatTrajectoryDistillation_FTD/distill/ema_logged_files/CIFAR100/decent-bird-4/ema_images_best.pt"
        if cifar=="CIFAR100":
            self.pt_dir = "/home/stulium/FlatTrajectoryDistillation_FTD/distill/logged_files/CIFAR100/decent-bird-4/images_5000.pt"
        # self.label_dir = "/home/stulium/FlatTrajectoryDistillation_FTD/distill/ema_logged_files/CIFAR100/decent-bird-4/ema_labels_best.pt"
            self.label_dir = "/home/stulium/FlatTrajectoryDistillation_FTD/distill/logged_files/CIFAR100/decent-bird-4/labels_5000.pt"
        else:
            self.pt_dir = "/home/stulium/FlatTrajectoryDistillation_FTD/distill/logged_files/CIFAR10/happy-silence-7/images_5000.pt"
            self.label_dir = "/home/stulium/FlatTrajectoryDistillation_FTD/distill/logged_files/CIFAR10/happy-silence-7/labels_5000.pt"
        images = torch.load(self.pt_dir)
        # zca = zca_op()
        # images = zca.inverse_transform(images)
        # images = normalize_minmax(images)
        dsa_param = ParamDiffAug()
        img = DiffAugment(images, 'color_crop_cutout_flip_scale_rotate', param=dsa_param)
        self.data = img
        self.labels = torch.load(self.label_dir)
        self.len = self.data.size(0)
        # self.transform = transform

    def __getitem__(self, idx):
        # transform_compose = transforms.Compose([transforms.ToPILImage(mode='RGB'), self.transform])
        # img_tensor = transform_compose(self.data[idx])
        img_tensor = self.data[idx]
        label = self.labels[idx].item()
        return img_tensor, label
    
    def __len__(self):
        return self.len
    
def normalize_minmax(tensor,scale_each=True):
    tensor = tensor.clone()  # avoid modifying tensor in-place
    # value_range = (float(t.min()), float(t.max()))
    def norm_ip(img, low, high):
        img.clamp_(min=low, max=high)
        img.sub_(low).div_(max(high - low, 1e-5))

    def norm_range(t):
        norm_ip(t, float(t.min()), float(t.max()))

    if scale_each is True:
        for t in tensor:  # loop over mini-batch dimension
            norm_range(t)
    else:
        norm_range(tensor)  
    return tensor

def zca_op():
    dst_train = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transforms.ToTensor())

    images = []
    labels = []
    print("Train ZCA")
    for i in tqdm.tqdm(range(len(dst_train))):
        im, lab = dst_train[i]
        images.append(im)
        labels.append(lab)
    images = torch.stack(images, dim=0).to('cpu')
    labels = torch.tensor(labels, dtype=torch.long, device="cpu")
    zca = K.enhance.ZCAWhitening(eps=0.1, compute_inv=True)
    zca.fit(images)
    return zca