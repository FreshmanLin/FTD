import sys
import os
import torch
import torchvision.transforms as transforms

load_dir = "ema_logged_files/CIFAR100/wandering-armadillo-2/ema_images_best.pt"
image=torch.load(load_dir)
print(image.size())