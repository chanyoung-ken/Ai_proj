import json
import os
import time
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from tqdm import tqdm

# AT+ALP utilities (assumed installed/available)
from attentionlpt import FeatureExtractor, compute_at_alp_loss
from attentionlpt import pgd_attack as alp_pgd_attack

# --------------------
# SE (Squeeze-and-Excitation) Module
# --------------------
class SELayer(nn.Module):
    def __init__(self, channel: int, reduction: int = 16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# --------------------
# CBAM (Convolutional Block Attention Module)
# --------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, channel: int, reduction: int = 16, kernel_size: int = 7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(channel, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.channel_att(x) * x
        out = self.spatial_att(out) * out
        return out

# --------------------
# ResNet18 with Parallel CBAM+SE Attention (Concat + 1x1 Conv merge)
# --------------------
class ResNet18MultiAttention(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(ResNet18MultiAttention, self).__init__()
        # Load pretrained ResNet18 backbone definition (for layers other than conv1/maxpool)
        backbone = models.resnet18(pretrained=False) # Use torchvision's definition

        # --- CIFAR10 Stem Modification ---
        # Original: 7x7 conv, stride 2, padding 3
        # Modified: 3x3 conv, stride 1, padding 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = backbone.bn1
        self.relu  = backbone.relu
        # self.maxpool = backbone.maxpool # Remove maxpool for CIFAR10
        # ------------------------------------

        # ResNet layers
        self.layer1 = backbone.layer1  # output channels: 64
        self.layer2 = backbone.layer2  # output channels: 128
        self.layer3 = backbone.layer3  # output channels: 256
        self.layer4 = backbone.layer4  # output channels: 512
        # Attention modules for each layer
        self.cbam1, self.se1 = CBAM(64), SELayer(64)
        self.merge1 = nn.Conv2d(64*2, 64, kernel_size=1, bias=False)
        self.bn_m1  = nn.BatchNorm2d(64)

        self.cbam2, self.se2 = CBAM(128), SELayer(128)
        self.merge2 = nn.Conv2d(128*2, 128, kernel_size=1, bias=False)
        self.bn_m2  = nn.BatchNorm2d(128)

        self.cbam3, self.se3 = CBAM(256), SELayer(256)
        self.merge3 = nn.Conv2d(256*2, 256, kernel_size=1, bias=False)
        self.bn_m3  = nn.BatchNorm2d(256)

        self.cbam4, self.se4 = CBAM(512), SELayer(512)
        self.merge4 = nn.Conv2d(512*2, 512, kernel_size=1, bias=False)
        self.bn_m4  = nn.BatchNorm2d(512)
        # Classifier head
        self.avgpool = backbone.avgpool
        self.fc      = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x) # Remove maxpool for CIFAR10

        # Layer1 + Parallel Attention + Merge
        f1 = self.layer1(x)
        a1_cbam = self.cbam1(f1)
        a1_se   = self.se1(f1)
        m1 = torch.cat([a1_cbam, a1_se], dim=1)
        m1 = self.bn_m1(self.merge1(m1)) # Apply BN after merge
        m1 = F.relu(m1) # Apply ReLU after BN
        # Layer2
        f2 = self.layer2(m1) # Pass merged features
        a2_cbam = self.cbam2(f2)
        a2_se   = self.se2(f2)
        m2 = torch.cat([a2_cbam, a2_se], dim=1)
        m2 = self.bn_m2(self.merge2(m2))
        m2 = F.relu(m2)
        # Layer3
        f3 = self.layer3(m2)
        a3_cbam = self.cbam3(f3)
        a3_se   = self.se3(f3)
        m3 = torch.cat([a3_cbam, a3_se], dim=1)
        m3 = self.bn_m3(self.merge3(m3))
        m3 = F.relu(m3)
        # Layer4
        f4 = self.layer4(m3)
        a4_cbam = self.cbam4(f4)
        a4_se   = self.se4(f4)
        m4 = torch.cat([a4_cbam, a4_se], dim=1)
        m4 = self.bn_m4(self.merge4(m4))
        m4 = F.relu(m4)
        # Classification
        out = self.avgpool(m4)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

# --------------------
# Hyperparameters (Load from JSON)
# --------------------
try:
    with open('best_hyperparams.json', 'r') as f:
        loaded_hp = json.load(f)
    # Use the correct key for the AT+ALP ResNet18 model trained for CIFAR10
    hp = loaded_hp["CIFAR10_AT_ALP_RESNET18"]
    print("Loaded hyperparameters from best_hyperparams.json")
except FileNotFoundError:
    print("Warning: best_hyperparams.json not found. Using default values.")
    # Fallback to default values if JSON is missing (optional, adjust as needed)
    hp = {
        "lr": 0.05,
        "wd": 0.005,
        "bs": 256,
        "opt": "SGD"
    }
except KeyError:
     print("Warning: Key 'CIFAR10_AT_ALP_RESNET18' not found in best_hyperparams.json. Using default values.")
     # Fallback if the specific key is missing
     hp = {
        "lr": 0.05,
        "wd": 0.005,
        "bs": 256,
        "opt": "SGD"
    }


BATCH_SIZE = hp.get("bs", 256) # Provide default if key missing
LR = hp.get("lr", 0.05)
WEIGHT_DECAY = hp.get("wd", 0.005)
OPTIMIZER = hp.get("opt", "SGD").upper() # 'SGD' or 'ADAMW'
EPOCHS = 50 # Keep EPOCHS or load if it's in JSON

# PGD attack settings for training
ADV_EPS = 8/255
ADV_ALPHA = 2/255
ADV_ITERS = 7

# Attention pairing weights
ALPHA_ALP = 1.0  # logit pairing weight
BETA_AT = 1.0    # attention map pairing weight
AT_LAYERS = ['layer1', 'layer2', 'layer3', 'layer4']

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --------------------
# Data Loaders
# --------------------
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# --------------------
# Model & Utilities
# --------------------
# Use ResNet18 pre-defined in torchvision, adjust for CIFAR10
model = ResNet18MultiAttention(num_classes=10)
model = model.to(device)

# Feature extractor for AT loss
feature_extractor = FeatureExtractor(model, AT_LAYERS)

# Optimizer
if OPTIMIZER == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY)
elif OPTIMIZER == 'ADAMW':
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
else:
    raise ValueError(f"Unsupported optimizer: {OPTIMIZER}")

# LR Scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# Loss placeholder (will be returned by compute_at_alp_loss)

# --------------------
# Training Loop
# --------------------
best_robust_acc = 0.0
best_weights = copy.deepcopy(model.state_dict())
start_time = time.time()

for epoch in range(1, EPOCHS+1):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # Compute AT+ALP loss
        attack_kwargs = {'eps': ADV_EPS, 'alpha': ADV_ALPHA, 'iters': ADV_ITERS}
        total_loss, ce_loss, alp_loss, at_loss = compute_at_alp_loss(
            model, feature_extractor,
            images, labels,
            alpha=ALPHA_ALP, beta=BETA_AT,
            attack_kwargs=attack_kwargs
        )

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        pbar.set_postfix({
            'Total': f"{total_loss.item():.3f}",
            'CE': f"{ce_loss.item():.3f}",
            'LP': f"{alp_loss.item():.3f}",
            'AT': f"{at_loss.item():.3f}"  
        })

    scheduler.step()

    # Evaluate robustness via simple PGD
    model.eval()
    correct, total = 0, 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        # 수정: alp_pgd_attack 호출 시 인자 순서 및 불필요 인자 제거
        adv_images = alp_pgd_attack(
            model=model,
            x=images,
            y=labels,
            eps=ADV_EPS,
            alpha=ADV_ALPHA,
            iters=ADV_ITERS
        )
        
        with torch.no_grad():
            outputs = model(adv_images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    robust_acc = correct / total

    print(f"Epoch {epoch}: Robust Acc (PGD eps={ADV_EPS:.3f}): {robust_acc:.4f}")
    if robust_acc > best_robust_acc:
        best_robust_acc = robust_acc
        best_weights = copy.deepcopy(model.state_dict())

# --------------------
# Save Best Model
# --------------------
model.load_state_dict(best_weights)
os.makedirs('saved_models', exist_ok=True)
model_path = os.path.join('saved_models', 'cifar10_at_alp_resnet18.pth')
torch.save(model.state_dict(), model_path)
print(f"Best robust acc: {best_robust_acc:.4f}, model saved to {model_path}")
print(f"Total training time: {time.time() - start_time:.1f}s")

# Cleanup hooks
feature_extractor.remove_hooks()
