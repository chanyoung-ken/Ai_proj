import os
import argparse
import json
import copy
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from pytorch_grad_cam import GradCAM

# --- Configuration ---
CONFIG_FILE = 'best_hyperparams.json'
RESULTS_DIR = 'robustness_results'
GRADCAM_DIR = 'gradcam_visualizations'
os.makedirs(GRADCAM_DIR, exist_ok=True)

# Dataset normalization
DATASET_MEANS = {'CIFAR10': (0.4914, 0.4822, 0.4465), 'CIFAR100': (0.5071, 0.4867, 0.4408)}
DATASET_STDS  = {'CIFAR10': (0.2023, 0.1994, 0.2010), 'CIFAR100': (0.2675, 0.2565, 0.2761)}

# Model architectures
def get_base_resnet(arch, num_classes):
    model = getattr(models, arch)(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

class ChannelAttention(nn.Module):
    def __init__(self, in_ch, reduction=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_ch, in_ch//reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_ch//reduction, in_ch, 1, bias=False)
        )
        self.sig = nn.Sigmoid()
    def forward(self, x):
        a = self.fc(self.avg(x))
        m = self.fc(self.max(x))
        return self.sig(a + m)

class SpatialAttention(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        pad = 3 if k==7 else 1
        self.conv = nn.Conv2d(2,1,k,padding=pad,bias=False)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        a = x.mean(1,keepdim=True)
        m,_ = x.max(1,keepdim=True)
        y = torch.cat([a,m],1)
        return self.sig(self.conv(y))

class CBAM(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ca = ChannelAttention(ch)
        self.sa = SpatialAttention()
    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class ResNet_CBAM(nn.Module):
    def __init__(self, arch, num_classes):
        super().__init__()
        base = get_base_resnet(arch, num_classes)
        self.conv1,self.bn1,self.relu,self.maxpool = base.conv1, base.bn1, base.relu, base.maxpool
        self.layer1,self.layer2,self.layer3,self.layer4 = base.layer1, base.layer2, base.layer3, base.layer4
        ch = [64,128,256,512]
        self.cbam1,self.cbam2,self.cbam3,self.cbam4 = CBAM(ch[0]),CBAM(ch[1]),CBAM(ch[2]),CBAM(ch[3])
        self.avgpool,self.fc = base.avgpool, base.fc
    def forward(self,x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.cbam1(self.layer1(x))
        x = self.cbam2(self.layer2(x))
        x = self.cbam3(self.layer3(x))
        x = self.cbam4(self.layer4(x))
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        return self.fc(x)

class ResNet_SE(nn.Module):
    def __init__(self, arch, num_classes):
        super().__init__()
        base = get_base_resnet(arch, num_classes)
        self.conv1,self.bn1,self.relu,self.maxpool = base.conv1, base.bn1, base.relu, base.maxpool
        self.layer1,self.layer2,self.layer3,self.layer4 = base.layer1, base.layer2, base.layer3, base.layer4
        ch = [64,128,256,512]
        self.se1 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(ch[0],ch[0]//16), nn.ReLU(), nn.Linear(ch[0]//16,ch[0]), nn.Sigmoid())
        self.se2 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(ch[1],ch[1]//16), nn.ReLU(), nn.Linear(ch[1]//16,ch[1]), nn.Sigmoid())
        self.se3 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(ch[2],ch[2]//16), nn.ReLU(), nn.Linear(ch[2]//16,ch[2]), nn.Sigmoid())
        self.se4 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(ch[3],ch[3]//16), nn.ReLU(), nn.Linear(ch[3]//16,ch[3]), nn.Sigmoid())
        self.avgpool,self.fc = base.avgpool, base.fc
    def forward(self,x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        b1 = self.layer1(x); x = b1 * self.se1(b1)
        b2 = self.layer2(x); x = b2 * self.se2(b2)
        b3 = self.layer3(x); x = b3 * self.se3(b3)
        b4 = self.layer4(x); x = b4 * self.se4(b4)
        x = self.avgpool(x); x = torch.flatten(x,1)
        return self.fc(x)

# Helper to unnormalize for visualization
def unnormalize(img, mean, std):
    img = img.cpu().numpy().transpose(1,2,0)
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)

# Load a single sample
def get_sample_image(ds_name, idx):
    mean, std = DATASET_MEANS[ds_name], DATASET_STDS[ds_name]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean,std)])
    Dataset = datasets.CIFAR10 if ds_name=='CIFAR10' else datasets.CIFAR100
    ds = Dataset('data', train=False, download=True, transform=transform)
    img, label = ds[idx]
    return img.unsqueeze(0), label

# Main
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['CIFAR10','CIFAR100'], required=True)
    parser.add_argument('--models', type=str, default='baseline,cbam,se,adv')
    parser.add_argument('--img-idx', type=int, default=0)
    args = parser.parse_args()

    # load hyperparams
    with open(CONFIG_FILE) as f:
        best = json.load(f)[args.dataset]
    arch = 'resnet18'
    num_classes = 10 if args.dataset=='CIFAR10' else 100

    # sample image
    img_tensor, label = get_sample_image(args.dataset, args.img_idx)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_tensor = img_tensor.to(device)

    # iterate models
    for key in args.models.split(','):
        if key=='baseline': m = get_base_resnet(arch,num_classes)
        elif key=='cbam': m = ResNet_CBAM(arch,num_classes)
        elif key=='se':   m = ResNet_SE(arch,num_classes)
        elif key=='adv':  m = get_base_resnet(arch,num_classes)
        else: continue
        # load weights
        path = os.path.join(RESULTS_DIR, f"{args.dataset}_{(key if key!='baseline' else 'baseline_resnet')}.pth")
        m.load_state_dict(torch.load(path, map_location='cpu'))
        m.to(device).eval()

        # GradCAM
        target_layer = m.layer4 if hasattr(m,'layer4') else m.layer3
        cam = GradCAM(model=m, target_layers=[target_layer])
        grayscale = cam(input_tensor=img_tensor)[0]
        orig = unnormalize(img_tensor[0], DATASET_MEANS[args.dataset], DATASET_STDS[args.dataset])
        heatmap = cv2.applyColorMap(np.uint8(255*grayscale), cv2.COLORMAP_JET)
        vis = cv2.addWeighted(orig,0.6,heatmap,0.4,0)

        fname = f"{args.dataset}_{key}_idx{args.img_idx}.png"
        cv2.imwrite(os.path.join(GRADCAM_DIR, fname), vis)
        print("Saved GradCAM:", fname)
