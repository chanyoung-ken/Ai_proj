# train_resnet.py
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models

# --- 설정 ---
TRAIN_DIR = '/home/work/AIprogramming/Ai_proj/data_generator/train_data'
TEST_DIR  = '/home/work/AIprogramming/Ai_proj/data_generator/test_data'
CHECKPOINT_DIR = './checkpoints'
HISTORY_CSV    = './history.csv'
BATCH_SIZE = 128
EPOCHS     = 30
LR         = 1e-3
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def load_pt_folder(pt_dir):
    """폴더의 .pt 파일 전부 읽어서 하나의 TensorDataset으로 반환"""
    imgs_list, labels_list = [], []
    for path in sorted(glob.glob(os.path.join(pt_dir, '*.pt'))):
        print(f'Loading {path} ...')
        imgs, lbls = torch.load(path)
        imgs_list.append(imgs)
        labels_list.append(lbls)
    imgs = torch.cat(imgs_list, dim=0)
    lbls = torch.cat(labels_list, dim=0)
    return TensorDataset(imgs, lbls)

# 1) 데이터 준비
train_ds = load_pt_folder(TRAIN_DIR)
test_ds  = load_pt_folder(TEST_DIR)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# 2) 모델 준비 (ResNet-18, CIFAR10용으로 마지막 레이어 수정)
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(DEVICE)

# 3) 손실/최적화 준비
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)

# 4) 학습 루프
history = []
best_acc = 0.0

for epoch in range(1, EPOCHS+1):
    # --- train ---
    model.train()
    running_loss = 0.0
    running_correct = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        running_correct += (outputs.argmax(1) == labels).sum().item()
    train_loss = running_loss / len(train_loader.dataset)
    train_acc  = running_correct / len(train_loader.dataset)

    # --- validate ---
    model.eval()
    val_loss = 0.0
    val_correct = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()
    val_loss /= len(test_loader.dataset)
    val_acc  = val_correct / len(test_loader.dataset)

    # --- 스케줄러 스텝 & 체크포인트 ---
    scheduler.step(val_acc)
    is_best = val_acc > best_acc
    if is_best:
        best_acc = val_acc
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'best_resnet18.pth'))

    # --- 히스토리 기록 ---
    print(f"Epoch {epoch:02d}/{EPOCHS} | "
          f"Train: loss={train_loss:.4f}, acc={train_acc:.4f} | "
          f"Val:   loss={val_loss:.4f}, acc={val_acc:.4f} {'*' if is_best else ''}")
    history.append({
        'epoch': epoch,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc
    })

# 5) 히스토리 저장
pd.DataFrame(history).to_csv(HISTORY_CSV, index=False)
print("Training complete. Best val_acc:", best_acc)
print("Checkpoint and history saved.")
