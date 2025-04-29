import json
import os
import time
import copy
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from tqdm import tqdm
import mlflow
import mlflow.pytorch

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
def parse_args():
    parser = argparse.ArgumentParser(description='Train ResNet18 with AT+ALP on CIFAR10')
    parser.add_argument('--adv_eps', type=float, default=8/255, 
                        help='PGD attack epsilon for training (default: 8/255)')
    parser.add_argument('--epochs', type=int, default=50, 
                        help='Number of training epochs (default: 50)')
    return parser.parse_args()

args = parse_args()

# MLflow 실험 이름 설정 (선택사항, 환경 변수로도 설정 가능)
# mlflow.set_experiment("CIFAR10_AT_ALP_ResNet18") # 주석 처리하거나 원하는 이름 사용

try:
    with open('best_hyperparams.json', 'r') as f:
        loaded_hp = json.load(f)
    # Use the correct key for the AT+ALP ResNet18 model trained for CIFAR10
    hp = loaded_hp["CIFAR10_AT_ALP_RESNET18"]
    print("Loaded hyperparameters from best_hyperparams.json")
except FileNotFoundError:
    print("Warning: best_hyperparams.json not found. Using default values.")
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
EPOCHS = args.epochs

# PGD attack settings for training
ADV_EPS = args.adv_eps
ADV_ALPHA = ADV_EPS / 4
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

# --------------------
# Training Loop with MLflow
# --------------------
best_robust_acc = 0.0
start_time = time.time()

# MLflow 실행 시작
with mlflow.start_run() as run:
    print(f"MLflow Run ID: {run.info.run_id}")
    # --- 파라미터 로깅 ---
    mlflow.log_param("adv_eps", ADV_EPS)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("learning_rate", LR)
    mlflow.log_param("weight_decay", WEIGHT_DECAY)
    mlflow.log_param("optimizer", OPTIMIZER)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("adv_alpha", ADV_ALPHA)
    mlflow.log_param("adv_iters", ADV_ITERS)
    mlflow.log_param("alpha_alp", ALPHA_ALP)
    mlflow.log_param("beta_at", BETA_AT)
    mlflow.log_param("at_layers", ",".join(AT_LAYERS)) # 리스트는 문자열로 변환하여 로깅

    best_weights = copy.deepcopy(model.state_dict()) # 베스트 가중치 초기화

    for epoch in range(1, EPOCHS+1):
        model.train()
        epoch_total_loss = 0.0
        epoch_ce_loss = 0.0
        epoch_alp_loss = 0.0
        epoch_at_loss = 0.0
        num_batches = 0

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

            # 에폭 누적 손실 계산
            epoch_total_loss += total_loss.item()
            epoch_ce_loss += ce_loss.item()
            # alp_loss는 매우 작을 수 있으므로 주의
            epoch_alp_loss += alp_loss.item() if alp_loss is not None else 0.0 
            epoch_at_loss += at_loss.item() if at_loss is not None else 0.0
            num_batches += 1
            
            # 배치별 로깅 (너무 많으면 주석 처리 가능)
            # mlflow.log_metric("batch_total_loss", total_loss.item())
            # mlflow.log_metric("batch_ce_loss", ce_loss.item())
            # mlflow.log_metric("batch_alp_loss", alp_loss.item() if alp_loss is not None else 0.0)
            # mlflow.log_metric("batch_at_loss", at_loss.item() if at_loss is not None else 0.0)

            pbar.set_postfix({
                'Total': f"{total_loss.item():.3f}",
                'CE': f"{ce_loss.item():.3f}",
                'LP': f"{alp_loss.item():.10f}" if alp_loss is not None else "N/A",
                'AT': f"{at_loss.item():.3f}" if at_loss is not None else "N/A"
            })
            
            # 기존 [DEBUG] 프린트 제거 또는 MLflow 로깅으로 대체 가능
            # print(f"[DEBUG] Raw loss values: CE={ce_loss.item():.6f}, LP={alp_loss.item():.10f}, AT={at_loss.item():.6f}")

        scheduler.step()

        # --- 에폭별 메트릭 로깅 ---
        avg_total_loss = epoch_total_loss / num_batches
        avg_ce_loss = epoch_ce_loss / num_batches
        avg_alp_loss = epoch_alp_loss / num_batches
        avg_at_loss = epoch_at_loss / num_batches
        
        mlflow.log_metric("avg_total_loss", avg_total_loss, step=epoch)
        mlflow.log_metric("avg_ce_loss", avg_ce_loss, step=epoch)
        mlflow.log_metric("avg_alp_loss", avg_alp_loss, step=epoch)
        mlflow.log_metric("avg_at_loss", avg_at_loss, step=epoch)
        mlflow.log_metric("learning_rate", scheduler.get_last_lr()[0], step=epoch) # 현재 학습률 로깅

        # Evaluate robustness via simple PGD
        model.eval()
        correct, total = 0, 0
        with torch.no_grad(): # 평가 시 그래디언트 계산 비활성화
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

                outputs = model(adv_images)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        robust_acc = correct / total

        # --- 강건 정확도 로깅 ---
        mlflow.log_metric("robust_acc", robust_acc, step=epoch)
        print(f"Epoch {epoch}: Robust Acc (PGD eps={ADV_EPS:.3f}): {robust_acc:.4f}")

        if robust_acc > best_robust_acc:
            best_robust_acc = robust_acc
            best_weights = copy.deepcopy(model.state_dict())
            # 최고 성능 갱신 시 로그 추가 (선택 사항)
            mlflow.log_metric("best_robust_acc_so_far", best_robust_acc, step=epoch) 
            print(f"*** New best robust accuracy: {best_robust_acc:.4f} at epoch {epoch} ***")


    # --- 최종 결과 및 모델 로깅 ---
    print(f"Training finished. Best Robust Accuracy: {best_robust_acc:.4f}")
    mlflow.log_metric("best_robust_acc", best_robust_acc) # 최종 최고 정확도 로깅

    # 최고 가중치로 모델 상태 복원
    model.load_state_dict(best_weights)

    # 모델 로컬 저장 (기존 방식 유지)
    os.makedirs('saved_models', exist_ok=True)
    eps_str = f"eps{ADV_EPS:.4f}".replace('.', '_')
    model_filename = f'cifar10_at_alp_resnet18_{eps_str}.pth'
    model_path = os.path.join('saved_models', model_filename)
    torch.save(model.state_dict(), model_path)
    print(f"Best model weights saved locally to {model_path}")

    # MLflow에 모델 아티팩트 로깅
    # 'model'은 MLflow UI에 표시될 아티팩트 내 폴더 이름입니다.
    # registered_model_name을 사용하면 모델 레지스트리에 등록할 수 있습니다.
    registered_model_name = f"cifar10-resnet18-at-alp-eps{eps_str}" # 모델 레지스트리 이름 (선택사항)
    print(f"Logging model to MLflow artifact path 'model'...")
    mlflow.pytorch.log_model(model, "model", registered_model_name=registered_model_name)
    print("Model logged to MLflow successfully.")
    
    # MLflow에 로컬 모델 파일도 아티팩트로 저장 (선택사항)
    # mlflow.log_artifact(model_path, artifact_path="local_saved_model")

# --------------------
# 스크립트 종료
# --------------------
total_training_time = time.time() - start_time
print(f"Total training time: {total_training_time:.1f}s")
# MLflow에 총 학습 시간 로깅 (run 컨텍스트 밖에서도 가능)
# mlflow.log_metric("total_training_time_seconds", total_training_time) # 주석 해제 시 마지막 run에 기록됨

# Cleanup hooks
feature_extractor.remove_hooks()
print("Feature extractor hooks removed.")

# FINAl_ROBUST_ACC 프린트 제거 (MLflow가 관리)
# print(f"FINAL_ROBUST_ACC:{best_robust_acc:.4f}")
