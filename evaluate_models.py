import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from tqdm import tqdm

# -------------------------------------
# 모델 아키텍처 (realensemble.py에서 복사)
# -------------------------------------
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

class ResNet18MultiAttention(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(ResNet18MultiAttention, self).__init__()
        backbone = models.resnet18(pretrained=False) # Load definition, not weights

        # --- CIFAR10 Stem Modification ---
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = backbone.bn1
        self.relu  = backbone.relu
        # No maxpool for CIFAR10

        # ResNet layers
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # Attention modules
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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        f1 = self.layer1(x)
        a1_cbam = self.cbam1(f1)
        a1_se   = self.se1(f1)
        m1 = torch.cat([a1_cbam, a1_se], dim=1)
        m1 = self.bn_m1(self.merge1(m1))
        m1 = F.relu(m1)

        f2 = self.layer2(m1)
        a2_cbam = self.cbam2(f2)
        a2_se   = self.se2(f2)
        m2 = torch.cat([a2_cbam, a2_se], dim=1)
        m2 = self.bn_m2(self.merge2(m2))
        m2 = F.relu(m2)

        f3 = self.layer3(m2)
        a3_cbam = self.cbam3(f3)
        a3_se   = self.se3(f3)
        m3 = torch.cat([a3_cbam, a3_se], dim=1)
        m3 = self.bn_m3(self.merge3(m3))
        m3 = F.relu(m3)

        f4 = self.layer4(m3)
        a4_cbam = self.cbam4(f4)
        a4_se   = self.se4(f4)
        m4 = torch.cat([a4_cbam, a4_se], dim=1)
        m4 = self.bn_m4(self.merge4(m4))
        m4 = F.relu(m4)

        out = self.avgpool(m4)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

# -------------------------------------
# 평가 함수
# -------------------------------------
def evaluate_model(model_path, test_loader, device, num_classes=10):
    """주어진 경로의 모델을 로드하고 테스트 데이터셋에서 평가합니다."""
    print(f"\n--- Evaluating Model: {os.path.basename(model_path)} ---")

    # 모델 초기화 및 가중치 로드
    model = ResNet18MultiAttention(num_classes=num_classes)
    try:
        # map_location을 사용하여 CPU에서도 로드 가능하게 함
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model weights from {model_path}: {e}")
        return None

    model.to(device)
    model.eval() # 평가 모드로 설정

    all_preds = []
    all_labels = []

    with torch.no_grad(): # 그래디언트 계산 비활성화
        for images, labels in tqdm(test_loader, desc="Predicting"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)

# -------------------------------------
# 메인 실행 로직
# -------------------------------------
def main(model_dir, data_dir, batch_size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # CIFAR-10 클래스 이름
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 데이터 변환 (realensemble.py와 동일하게 설정)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    # 테스트 데이터셋 로드
    try:
        test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    except Exception as e:
        print(f"Error loading CIFAR10 dataset from {data_dir}: {e}")
        return

    # 모델 디렉토리에서 .pth 파일 찾기
    model_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith('.pth')]

    if not model_files:
        print(f"No '.pth' model files found in directory: {model_dir}")
        return

    for model_path in model_files:
        results = evaluate_model(model_path, test_loader, device, num_classes=len(classes))

        if results is not None:
            true_labels, pred_labels = results

            # 전체 정확도
            accuracy = accuracy_score(true_labels, pred_labels)
            print(f"Overall Accuracy: {accuracy:.4f}")

            # 분류 보고서 (정밀도, 재현율, F1-점수)
            print("\nClassification Report:")
            print(classification_report(true_labels, pred_labels, target_names=classes, digits=4))

            # 혼동 행렬 계산
            cm = confusion_matrix(true_labels, pred_labels)

            # 혼동 행렬 시각화
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title(f'Confusion Matrix - {os.path.basename(model_path)}')
            plt.tight_layout()
            # 이미지 파일로 저장 (선택 사항)
            # plt.savefig(os.path.join(model_dir, f"{os.path.splitext(os.path.basename(model_path))[0]}_confusion_matrix.png"))
            plt.show() # 화면에 표시

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained ResNet18 models on CIFAR10.')
    # 사용자의 실제 경로를 기본값으로 설정
    parser.add_argument('--model_dir', type=str, default='C:/Users/chany/coding/Ai_proj/saved_models',
                        help='Directory containing the saved model (.pth) files.')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory where CIFAR10 data is stored or will be downloaded.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for evaluation.')

    args = parser.parse_args()

    # 모델 디렉토리가 존재하는지 확인
    if not os.path.isdir(args.model_dir):
        print(f"Error: Model directory not found at {args.model_dir}")
    else:
        main(args.model_dir, args.data_dir, args.batch_size)
