import os
import json
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import numpy as np
import pandas as pd
from torchvision import datasets, transforms, models
from tqdm import tqdm
import time
import matplotlib.pyplot as plt


# --- 설정 ---
CONFIG_FILE = 'best_hyperparams.json'
DATASETS = ['CIFAR10', 'CIFAR100']
DATA_DIR = './data'
BATCH_SIZE = 128
FINAL_EPOCHS = 100
ADV_EPS = 8/255
ADV_ALPHA = 2/255
ADV_ITERS = 7
USE_SUBSET = False
SUBSET_SIZE = 1000
RESULTS_DIR = 'robustness_results'
BASE_MODEL_ARCH = 'resnet18'
OPTIMIZER_TYPE  = 'SGD'         # ← 이 줄을 추가!

os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Helper functions (data loading, model setup, attacks, evaluation) ---

# Dataset normalization parameters
DATASET_MEANS = {
    'CIFAR10': (0.4914, 0.4822, 0.4465),
    'CIFAR100': (0.5071, 0.4867, 0.4408)
}
DATASET_STDS = {
    'CIFAR10': (0.2023, 0.1994, 0.2010),
    'CIFAR100': (0.2675, 0.2565, 0.2761)
}


def get_data_loader(name, batch_size, train=True):
    """
    Create DataLoader for CIFAR10 or CIFAR100.
    """
    mean = DATASET_MEANS[name]
    std = DATASET_STDS[name]
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
        dataset = datasets.CIFAR10(DATA_DIR, train=train, download=True, transform=transform)
        num_classes = 10
    elif name == 'CIFAR100':
        dataset = datasets.CIFAR100(DATA_DIR, train=train, download=True, transform=transform)
        num_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {name}")
    if USE_SUBSET:
        indices = np.random.choice(len(dataset), SUBSET_SIZE, replace=False)
        dataset = Subset(dataset, indices)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=2, pin_memory=True)
    return loader, num_classes


def get_base_resnet(arch, num_classes):
    """
    Return a ResNet model adapted for CIFAR datasets.
    """
    # Load without pretrained weights
    model = getattr(models, arch)(pretrained=False)
    # Adjust first conv layer and remove maxpool for CIFAR
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# Attention modules for CBAM
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x

# SE module
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResNet_CBAM(nn.Module):
    def __init__(self, base_arch, num_classes):
        super().__init__()
        base = get_base_resnet(base_arch, num_classes)
        # Copy initial layers
        self.conv1, self.bn1, self.relu, self.maxpool = base.conv1, base.bn1, base.relu, base.maxpool
        # Layers
        self.layer1, self.layer2, self.layer3, self.layer4 = base.layer1, base.layer2, base.layer3, base.layer4
        # CBAM modules
        channels = [64, 128, 256, 512]
        self.cbam1 = CBAM(channels[0])
        self.cbam2 = CBAM(channels[1])
        self.cbam3 = CBAM(channels[2])
        self.cbam4 = CBAM(channels[3])
        self.avgpool = base.avgpool
        self.fc = base.fc

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.cbam1(self.layer1(x))
        x = self.cbam2(self.layer2(x))
        x = self.cbam3(self.layer3(x))
        x = self.cbam4(self.layer4(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

class ResNet_SE(nn.Module):
    def __init__(self, base_arch, num_classes):
        super().__init__()
        base = get_base_resnet(base_arch, num_classes)
        self.conv1, self.bn1, self.relu, self.maxpool = base.conv1, base.bn1, base.relu, base.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = base.layer1, base.layer2, base.layer3, base.layer4
        channels = [64, 128, 256, 512]
        self.se1 = SELayer(channels[0])
        self.se2 = SELayer(channels[1])
        self.se3 = SELayer(channels[2])
        self.se4 = SELayer(channels[3])
        self.avgpool = base.avgpool
        self.fc = base.fc

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.se1(self.layer1(x))
        x = self.se2(self.layer2(x))
        x = self.se3(self.layer3(x))
        x = self.se4(self.layer4(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# Adversarial attacks
def fgsm_attack(model, criterion, images, labels, epsilon, device):
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    images.requires_grad = True
    outputs = model(images)
    loss = criterion(outputs, labels)
    model.zero_grad()
    loss.backward()
    perturbation = epsilon * images.grad.sign()
    perturbed = torch.clamp(images + perturbation, 0, 1)
    return perturbed.detach()


def pgd_attack(model, criterion, images, labels, epsilon, alpha, iters, device):
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    original = images.clone().detach()
    for _ in range(iters):
        images.requires_grad = True
        outputs = model(images)
        loss = criterion(outputs, labels)
        model.zero_grad()
        loss.backward()
        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - original, -epsilon, epsilon)
        images = torch.clamp(original + eta, 0, 1).detach()
    return images


def evaluate(model, device, loader, attack=None, attack_name=None, **attack_params):
    model.eval()
    correct, total = 0, 0
    criterion = nn.CrossEntropyLoss()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if attack:
            images = attack(model, criterion, images, labels, device=device, **attack_params)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def robustness_benchmark(model, device, test_loader, attack_type):
    results = {}
    if attack_type == 'fgsm':
        eps_values = [0.01, 0.03, 0.05, 0.07, 0.1]
        for eps in eps_values:
            acc = evaluate(model, device, test_loader, attack=fgsm_attack, attack_name='FGSM', epsilon=eps)
            results[eps] = acc
            print(f"FGSM attack (eps={eps}): accuracy={acc:.4f}")
    elif attack_type == 'pgd':
        eps = ADV_EPS
        alpha = ADV_ALPHA
        for iters in [5, 10, 20]:
            acc = evaluate(model, device, test_loader, attack=pgd_attack, attack_name='PGD', epsilon=eps, alpha=alpha, iters=iters)
            results[iters] = acc
            print(f"PGD attack (iters={iters}, eps={eps}): accuracy={acc:.4f}")
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")
    return results

# Training loops
def train_final_model(model, device, train_loader, test_loader, epochs, lr, weight_decay, optimizer_type, model_name="Model", dataset_name="Dataset"):
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay) if optimizer_type.lower() == 'sgd' else optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0
    best_weights = copy.deepcopy(model.state_dict())
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        for images, labels in tqdm(train_loader, desc=f"Training {model_name} epoch {epoch+1}/{epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        acc = evaluate(model, device, test_loader)
        print(f"Epoch {epoch+1}/{epochs} - {model_name} validation accuracy: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())
    total_time = time.time() - start_time
    model.load_state_dict(best_weights)
    print(f"Best validation accuracy for {model_name} on {dataset_name}: {best_acc:.4f}")
    return model, best_acc, total_time


def train_model_adversarial(model, device, train_loader, test_loader, epochs, lr, weight_decay, optimizer_type, attack_epsilon, attack_alpha, attack_iters, model_name="AdvModel", dataset_name="Dataset"):
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay) if optimizer_type.lower() == 'sgd' else optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    best_robust_acc = 0
    best_weights = copy.deepcopy(model.state_dict())
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        for images, labels in tqdm(train_loader, desc=f"Adv Training {model_name} epoch {epoch+1}/{epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)
            model.eval()
            adv_images = pgd_attack(model, criterion, images.detach(), labels, epsilon=attack_epsilon, alpha=attack_alpha, iters=attack_iters, device=device)
            model.train()
            optimizer.zero_grad()
            outputs = model(adv_images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        robust_acc = evaluate(model, device, test_loader, attack=pgd_attack, attack_name='PGD', epsilon=attack_epsilon, alpha=attack_alpha, iters=attack_iters)
        print(f"Epoch {epoch+1}/{epochs} - {model_name} robust accuracy: {robust_acc:.4f}")
        if robust_acc > best_robust_acc:
            best_robust_acc = robust_acc
            best_weights = copy.deepcopy(model.state_dict())
    total_time = time.time() - start_time
    model.load_state_dict(best_weights)
    print(f"Best robust accuracy for {model_name} on {dataset_name}: {best_robust_acc:.4f}")
    return model, best_robust_acc, total_time

# --- Main execution ---
def main():
    # Load best hyperparameters
    with open(CONFIG_FILE) as f:
        best_params = json.load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for ds_name in DATASETS:
        cfg = best_params.get(ds_name, {})
        lr = cfg.get('lr')
        wd = cfg.get('wd')
        bs = cfg.get('bs', BATCH_SIZE)
        opt = cfg.get('opt', OPTIMIZER_TYPE)
        print(f"[BENCHMARK] Dataset: {ds_name}, LR={lr}, WeightDecay={wd},{bs}{opt}")

        train_loader, num_classes = get_data_loader(ds_name, bs, train=True)
        test_loader, _            = get_data_loader(ds_name, bs, train=False)

        # 출력에도 batch size 같이 찍어 주기
        print(f"Dataset: {ds_name}, Batch size: {bs}, "
            f"Number of classes: {num_classes}, Epochs: {FINAL_EPOCHS}")

        # optimizer 설정도 opt 기반으로
        model     = get_base_resnet(BASE_MODEL_ARCH, num_classes).to(device)
        optimizer = (
            optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
            if opt=='SGD'
            else optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        )
        trained_models = {}
        training_times = {}

        # Baseline
        print(f"--- Training baseline ResNet on {ds_name} ---")
        base_model = get_base_resnet(BASE_MODEL_ARCH, num_classes)
        base_model, base_acc, base_time = train_final_model(base_model, device, train_loader, test_loader, FINAL_EPOCHS, lr, wd, OPTIMIZER_TYPE, model_name="BaselineResNet", dataset_name=ds_name)
        torch.save(base_model.state_dict(), os.path.join(RESULTS_DIR, f"{ds_name}_baseline_resnet.pth"))
        trained_models['BaselineResNet'] = copy.deepcopy(base_model.to('cpu'))
        training_times['BaselineResNet'] = base_time

        # CBAM
        print(f"--- Training ResNet+CBAM on {ds_name} ---")
        cbam_model = ResNet_CBAM(BASE_MODEL_ARCH, num_classes)
        cbam_model, cbam_acc, cbam_time = train_final_model(cbam_model, device, train_loader, test_loader, FINAL_EPOCHS, lr, wd, OPTIMIZER_TYPE, model_name="ResNetCBAM", dataset_name=ds_name)
        torch.save(cbam_model.state_dict(), os.path.join(RESULTS_DIR, f"{ds_name}_resnet_cbam.pth"))
        trained_models['ResNetCBAM'] = copy.deepcopy(cbam_model.to('cpu'))
        training_times['ResNetCBAM'] = cbam_time

        # SE
        print(f"--- Training ResNet+SE on {ds_name} ---")
        se_model = ResNet_SE(BASE_MODEL_ARCH, num_classes)
        se_model, se_acc, se_time = train_final_model(se_model, device, train_loader, test_loader, FINAL_EPOCHS, lr, wd, OPTIMIZER_TYPE, model_name="ResNetSE", dataset_name=ds_name)
        torch.save(se_model.state_dict(), os.path.join(RESULTS_DIR, f"{ds_name}_resnet_se.pth"))
        trained_models['ResNetSE'] = copy.deepcopy(se_model.to('cpu'))
        training_times['ResNetSE'] = se_time

        # Adversarial Training
        print(f"--- Adversarial training for baseline ResNet on {ds_name} ---")
        adv_model = get_base_resnet(BASE_MODEL_ARCH, num_classes)
        adv_model, adv_acc, adv_time = train_model_adversarial(adv_model, device, train_loader, test_loader, FINAL_EPOCHS, lr, wd, OPTIMIZER_TYPE, ADV_EPS, ADV_ALPHA, ADV_ITERS, model_name="AdvResNet", dataset_name=ds_name)
        torch.save(adv_model.state_dict(), os.path.join(RESULTS_DIR, f"{ds_name}_adv_resnet.pth"))
        trained_models['AdvResNet'] = copy.deepcopy(adv_model.to('cpu'))
        training_times['AdvResNet'] = adv_time

        # Evaluation
        print(f"--- Evaluating models on {ds_name} ---")
        results_clean = {}
        results_fgsm = {}
        results_pgd = {}
        for name, model_instance in trained_models.items():
            model_instance = model_instance.to(device)
            clean_acc = evaluate(model_instance, device, test_loader)
            print(f"{name} clean accuracy: {clean_acc:.4f}")
            results_clean[name] = clean_acc
            results_fgsm[name] = robustness_benchmark(model_instance, device, test_loader, 'fgsm')
            results_pgd[name] = robustness_benchmark(model_instance, device, test_loader, 'pgd')
            model_instance.to('cpu')

        # Save results to DataFrame and CSV
        df_clean = pd.DataFrame.from_dict(results_clean, orient='index', columns=['CleanAccuracy'])
        df_fgsm = pd.DataFrame(results_fgsm)
        df_pgd = pd.DataFrame(results_pgd)
        df_times = pd.DataFrame.from_dict(training_times, orient='index', columns=['TrainingTimeSeconds'])

        df_clean.to_csv(os.path.join(RESULTS_DIR, f"{ds_name}_clean_accuracy.csv"))
        df_fgsm.to_csv(os.path.join(RESULTS_DIR, f"{ds_name}_fgsm_robustness.csv"))
        df_pgd.to_csv(os.path.join(RESULTS_DIR, f"{ds_name}_pgd_robustness.csv"))
        df_times.to_csv(os.path.join(RESULTS_DIR, f"{ds_name}_training_times.csv"))

        # Plotting
        plt.style.use('seaborn-whitegrid')
        # FGSM plot
        plt.figure(figsize=(8,6))
        for col in df_fgsm.columns:
            eps = list(df_fgsm.index)
            accs = df_fgsm[col].values
            plt.plot(eps, accs, marker='o', label=col)
        plt.title(f"{ds_name} FGSM Robustness")
        plt.xlabel('Epsilon')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(os.path.join(RESULTS_DIR, f"{ds_name}_fgsm_plot.png"))
        plt.close()

        # PGD plot
        plt.figure(figsize=(8,6))
        for col in df_pgd.columns:
            iters = list(df_pgd.index)
            accs = df_pgd[col].values
            plt.plot(iters, accs, marker='s', linestyle='--', label=col)
        plt.title(f"{ds_name} PGD Robustness (eps={ADV_EPS})")
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(os.path.join(RESULTS_DIR, f"{ds_name}_pgd_plot.png"))
        plt.close()

    print("Benchmark completed.")

if __name__ == '__main__':
    main()

def main():
    # 설정 불러오기
    with open(CONFIG_FILE) as f:
        best_params = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for ds_name in DATASETS:
        cfg = best_params.get(ds_name, {})
        lr = cfg.get('lr', 0.01)
        wd = cfg.get('wd', 5e-4)
        print(f"[BENCH] {ds_name}: lr={lr}, wd={wd}")

        # 데이터 로더
        train_loader, num_classes = get_data_loader(ds_name, BATCH_SIZE, train=True)
        test_loader, _ = get_data_loader(ds_name, BATCH_SIZE, train=False)

        # 모델 훈련 및 평가 (baseline, CBAM, SE, adversarial)
        # ... Phase 2 코드 루프 복사

        # 결과 저장 (CSV, PNG)
        # ... 동일 처리

    print("벤치마크 완료.")

if __name__ == '__main__':
    main()
