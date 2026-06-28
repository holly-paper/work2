import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os, glob, sys, time

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_DIR = r"D:\作业\train"
TEST_DIR  = r"D:\作业\test"

def fmt_time(sec):
    if sec < 60: return f"{sec:.0f}s"
    return f"{int(sec)//60}m{int(sec)%60:02d}s"

# ===================== 1. 数据 =====================
print("=" * 55)
print("  CNN 第一层卷积核可视化")
print(f"  设备: {DEV.__str__().upper()}")
print("=" * 55)

class CIFAR10Folder(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder; self.transform = transform
        self.files = sorted(glob.glob(os.path.join(folder, '*.jpg')))
        if not self.files: self.files = sorted(glob.glob(os.path.join(folder, '*.png')))
        self.labels = [int(os.path.splitext(os.path.basename(f))[0].split('_')[0]) for f in self.files]
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, self.labels[idx]

print("\n[1/3] 加载数据...")
sys.stdout.flush()

tf_train = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                       T.ToTensor(), T.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
tf_test  = T.Compose([T.ToTensor(), T.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])

train_full = CIFAR10Folder(TRAIN_DIR, transform=tf_train)
testset    = CIFAR10Folder(TEST_DIR,  transform=tf_test)
train_size = int(0.9 * len(train_full))
trainset, _ = random_split(train_full, [train_size, len(train_full)-train_size],
                           generator=torch.Generator().manual_seed(42))
tr_loader = DataLoader(trainset, batch_size=64, shuffle=True,  num_workers=0)
te_loader = DataLoader(testset,  batch_size=64, shuffle=False, num_workers=0)
print(f"  训练: {train_size:,}  测试: {len(testset):,}")

# ===================== 2. 模型 =====================
print("\n[2/3] 训练 CNN (25 epochs)...")
sys.stdout.flush()

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(3, 32, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=1);  self.pool1 = nn.MaxPool2d(2)
        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=1);  self.conv2_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=1); self.conv3_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2)
        self.cls = nn.Sequential(nn.Flatten(), nn.Linear(128*4*4,256), nn.ReLU(),
                                  nn.Dropout(0.5), nn.Linear(256,10))
    def forward(self, x):
        x = self.relu(self.conv1_1(x)); x = self.relu(self.conv1_2(x)); x = self.pool1(x)
        x = self.relu(self.conv2_1(x)); x = self.relu(self.conv2_2(x)); x = self.pool2(x)
        x = self.relu(self.conv3_1(x)); x = self.relu(self.conv3_2(x)); x = self.pool3(x)
        return self.cls(x)

model = CNN().to(DEV)
crit = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=1e-3)
sched = optim.lr_scheduler.StepLR(opt, 15, 0.5)
t0 = time.time()

for ep in range(1, 26):
    model.train()
    for x, y in tr_loader: x, y = x.to(DEV), y.to(DEV); opt.zero_grad(); crit(model(x), y).backward(); opt.step()
    sched.step()
    if ep % 5 == 0 or ep == 1:
        model.eval()
        with torch.no_grad():
            acc = sum((model(x.to(DEV)).argmax(1)==y.to(DEV)).sum().item()
                      for x, y in te_loader) / len(testset) * 100
        elapsed = time.time() - t0
        eta = (elapsed/ep)*(25-ep) if ep>0 else 0
        n = int(ep/25*20); bar = "[" + "="*n + ">" + " "*(20-n-1) + "]"
        print(f"  {bar} {ep:2d}/25 | Test Acc: {acc:.1f}% | {fmt_time(elapsed):>5s} | ETA: {fmt_time(eta)}")
        sys.stdout.flush()

print(f"  训练完成!  总耗时: {fmt_time(time.time()-t0)}")

# ===================== 3. 卷积核可视化 =====================
print("\n[3/3] 提取 conv1_1 权重并可视化...")
sys.stdout.flush()

weights = model.conv1_1.weight.detach().cpu().numpy()  # (32, 3, 3, 3)
vmin, vmax = weights.min(), weights.max()
w_norm = (weights - vmin) / (vmax - vmin)  # 全局归一化到 [0,1]

print(f"  权重范围: [{vmin:.4f}, {vmax:.4f}]")
print(f"  归一化后:  [0, 1]")

fig, axes = plt.subplots(4, 8, figsize=(16, 9))
axes = axes.flatten()

for i in range(32):
    # (3, 3, 3) → transpose to (H, W, C) for imshow
    kernel_rgb = np.transpose(w_norm[i], (1, 2, 0))  # (3,3,3) → (3,3,3)
    axes[i].imshow(kernel_rgb, interpolation='nearest')
    axes[i].axis('off')
    axes[i].set_title(f'#{i+1}', fontsize=8)

plt.suptitle('CNN First Layer Convolution Kernels (32 filters, 3×3×3 RGB)',
             fontsize=14, fontweight='bold')
plt.tight_layout()

out = os.path.join(os.path.expanduser("~"), 'Desktop', 'conv_filters.png')
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f"\n  图片已保存: {out}")
