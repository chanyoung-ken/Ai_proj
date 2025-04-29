import os
import torch
import torchvision
import torchvision.transforms as T
import torchattacks
from tqdm import tqdm

# 저장할 공격 리스트
attacks = {
    'clean':        None,
    'fgsm':         lambda model: torchattacks.FGSM(model, eps=8/255),
    'pgd':          lambda model: torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=40, random_start=True),
    'rfgsm':        lambda model: torchattacks.RFGSM(model, eps=8/255, alpha=2/255),
    'mifgsm':       lambda model: torchattacks.MIFGSM(model, eps=8/255, steps=40, decay=1.0),
    'cw':           lambda model: torchattacks.CW(model, c=1, steps=1000, lr=0.01),
}

# 1) 데이터셋 로드 (train or test 중 선택)
transform = T.Compose([T.ToTensor(),])  # 나중에 공격 시 normalization은 atk.set_normalization_used로
dataset = torchvision.datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=True,
                                       transform=transform)
loader  = torch.utils.data.DataLoader(dataset,
                                      batch_size=256,
                                      shuffle=False,
                                      num_workers=4)

# 2) 모델 준비 (평가 모드, GPU)
model = torchvision.models.resnet50(pretrained=True).eval().cuda()

# 3) 출력 디렉터리 생성
os.makedirs('data', exist_ok=True)

# 4) 각 공격별로 이미지·레이블 생성 & 저장
for name, atk_fn in attacks.items():
    all_imgs = []
    all_lbls = []

    # (clean)은 atk_fn=None이므로 그냥 원본 저장
    if atk_fn is not None:
        atk = atk_fn(model)
        # CIFAR-10 전처리(mean/std) 필요 시
        atk.set_normalization_used(mean=[0.4914,0.4822,0.4465],
                                  std=[0.2471,0.2435,0.2616])
    else:
        atk = None

    print(f'==> Generating {name} examples...')
    for images, labels in tqdm(loader):
        images = images.cuda()
        labels = labels.cuda()

        if atk is not None:
            with torch.no_grad():
                adv_images = atk(images, labels)
        else:
            adv_images = images

        # CPU로 옮겨 담기
        all_imgs.append(adv_images.cpu())
        all_lbls.append(labels.cpu())

    # 하나의 큰 Tensor로 합치기
    imgs_tensor = torch.cat(all_imgs, dim=0)   # [N,3,32,32]
    lbls_tensor = torch.cat(all_lbls, dim=0)   # [N]

    # 저장
    save_path = f'data/{name}.pt'
    print(f'    saving {imgs_tensor.shape} to {save_path}')
    torch.save((imgs_tensor, lbls_tensor), save_path)

print('All done!')
