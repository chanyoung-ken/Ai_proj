
import os
import json
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset

# --- 설정 ---
HPO_EPOCHS = 10
BATCH_SIZE_LIST   = [64, 128, 256]
OPTIMIZER_LIST    = ['SGD', 'AdamW']
LR_LIST = [0.05]
WD_LIST = [0.005, 0.001]
DATASETS = ['CIFAR10', 'CIFAR100']
DATA_DIR = './data'
RESULT_FILE = 'best_hyperparams.json'
USE_SUBSET = False
SUBSET_SIZE = 1000

# --- 헬퍼 함수 ---
DATASET_MEANS = {'CIFAR10': (0.4914, 0.4822, 0.4465), 'CIFAR100': (0.5071, 0.4867, 0.4408)}
DATASET_STDS  = {'CIFAR10': (0.2023, 0.1994, 0.2010), 'CIFAR100': (0.2675, 0.2565, 0.2761)}

def get_data_loader(name, batch_size, train=True):
    mean, std = DATASET_MEANS[name], DATASET_STDS[name]
    if train:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    if name == 'CIFAR10':
        ds = datasets.CIFAR10(DATA_DIR, train=train, download=True, transform=transform)
        num_classes = 10
    else:
        ds = datasets.CIFAR100(DATA_DIR, train=train, download=True, transform=transform)
        num_classes = 100
    if USE_SUBSET:
        idx = np.random.choice(len(ds), SUBSET_SIZE, replace=False)
        ds = Subset(ds, idx)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=2, pin_memory=True)
    return loader, num_classes


def get_base_resnet(arch='resnet18', num_classes=10):
    model = getattr(models, arch)(pretrained=False)
    # CIFAR용 conv1 수정
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

# --- 그리드 서치 ---
def main():
    best_params = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for ds_name in DATASETS:
        _, num_classes = get_data_loader(ds_name, BATCH_SIZE_LIST[0], train=True)
        best_acc = 0.0
        best_cfg = {'lr': None, 'wd': None, 'bs': None, 'opt': None}

        for lr, wd, bs, opt in itertools.product(LR_LIST, WD_LIST, BATCH_SIZE_LIST, OPTIMIZER_LIST):
            print(f"[HPO] {ds_name} START|batch_size={bs},lr={lr}, wd={wd}" )

            train_loader, _ = get_data_loader(ds_name, bs, train=True)
            test_loader, _  = get_data_loader(ds_name, bs, train=False)
            model = get_base_resnet('resnet18', num_classes).to(device)
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)\
                        if opt=='SGD' else optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=HPO_EPOCHS)
            criterion = nn.CrossEntropyLoss()

            for epoch in range(HPO_EPOCHS):
                train_one_epoch(model, train_loader, optimizer, criterion, device)
                scheduler.step()

            acc = evaluate(model, test_loader, device)
            print(
                f"  bs={bs}, opt={opt}, "
                f"lr={lr:.4f}, wd={wd:.4f} -> "
                f"Test Acc: {acc:.4f}"
            )
            if acc > best_acc:
                best_acc = acc
                best_cfg = {'lr':lr,'wd':wd,'bs':bs,'opt':opt}

        best_params[ds_name] = {'best_acc': best_acc, **best_cfg}
    print(f"[HPO] {ds_name} 완료: best_acc={best_acc:.4f}, "
    f"lr={best_cfg['lr']}, wd={best_cfg['wd']}, "
    f"bs={best_cfg['bs']}, opt={best_cfg['opt']}")
    # 결과 저장
    with open(RESULT_FILE, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"HPO 결과가 '{RESULT_FILE}' 에 저장되었습니다.")

if __name__ == '__main__':
    main()