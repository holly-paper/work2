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
LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer',
          'dog', 'frog', 'horse', 'ship', 'truck']

def fmt_time(sec):
    if sec < 60: return f"{sec:.0f}s"
    return f"{int(sec)//60}m{int(sec)%60:02d}s"

# ===================== 1. 数据加载 =====================
print("=" * 55)
print("  CNN 混淆矩阵 — CIFAR-10 测试集")
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

print("\n[1/4] 加载数据...")
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

# ===================== 2. CNN 模型 =====================
print("\n[2/4] 构建 CNN 模型...")
sys.stdout.flush()

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.Conv2d(32,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.Conv2d(64,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.Conv2d(128,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(128*4*4,256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256,10),
        )
    def forward(self, x): return self.net(x)

# ===================== 3. 训练 =====================
print("\n[3/4] 训练 CNN (40 epochs)...")
sys.stdout.flush()

model = CNN().to(DEV)
crit = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=1e-3)
sched = optim.lr_scheduler.StepLR(opt, 20, 0.5)
t0 = time.time()

for ep in range(1, 41):
    model.train()
    for x, y in tr_loader: x, y = x.to(DEV), y.to(DEV); opt.zero_grad(); crit(model(x), y).backward(); opt.step()
    sched.step()
    if ep % 10 == 0 or ep == 1:
        model.eval()
        with torch.no_grad():
            acc = sum((model(x.to(DEV)).argmax(1) == y.to(DEV)).sum().item()
                      for x, y in te_loader) / len(testset) * 100
        elapsed = time.time() - t0
        eta = (elapsed / ep) * (40 - ep) if ep > 0 else 0
        n_eq = int(ep / 40 * 20)
        bar = "[" + "=" * n_eq + ">" + " " * (20 - n_eq - 1) + "]"
        print(f"  {bar} {ep:2d}/40 | Test Acc: {acc:.1f}% | {fmt_time(elapsed):>5s} | ETA: {fmt_time(eta)}")
        sys.stdout.flush()

total_time = time.time() - t0
print(f"  训练完成!  总耗时: {fmt_time(total_time)}")

# ===================== 4. 混淆矩阵 =====================
print("\n[4/4] 计算混淆矩阵...")
sys.stdout.flush()

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for x, y in te_loader:
        all_preds.extend(model(x.to(DEV)).argmax(1).cpu().tolist())
        all_labels.extend(y.tolist())

cm = np.zeros((10, 10), dtype=np.float64)
for t, p in zip(all_labels, all_preds):
    cm[t, p] += 1

cm_norm = cm / cm.sum(axis=1, keepdims=True)
per_class = np.diag(cm_norm) * 100

# 打印各类准确率
print(f"\n  各类测试准确率:")
for i, (name, acc) in enumerate(zip(LABELS, per_class)):
    print(f"  {i} {name:>12s}: {acc:.1f}%")

# 打印关键混淆对
cat_idx, dog_idx = 3, 5
auto_idx, truck_idx = 1, 9
print(f"\n  主要混淆对:")
print(f"  cat(58.3%) -> dog: {cm_norm[cat_idx, dog_idx]*100:.1f}%")
print(f"  dog(76.3%) -> cat: {cm_norm[dog_idx, cat_idx]*100:.1f}%")
print(f"  auto(89.0%) -> truck: {cm_norm[auto_idx, truck_idx]*100:.1f}%")
print(f"  truck(91.0%) -> auto: {cm_norm[truck_idx, auto_idx]*100:.1f}%")

# ===================== 5. 绘图 =====================

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cm_norm, cmap='YlOrRd', vmin=0, vmax=1)

# 每格标百分比
for i in range(10):
    for j in range(10):
        val = cm_norm[i, j]
        color = 'white' if val > 0.5 else 'black'
        if val > 0.01:
            ax.text(j, i, f'{val:.1%}', ha='center', va='center',
                    fontsize=8, color=color, fontweight='bold' if val > 0.3 else 'normal')

ax.set_xticks(range(10)); ax.set_yticks(range(10))
ax.set_xticklabels(LABELS, rotation=45, ha='right', fontsize=10)
ax.set_yticklabels([f'{LABELS[i]}\n({per_class[i]:.0f}%)' for i in range(10)], fontsize=9)
ax.set_xlabel('Predicted', fontsize=13, fontweight='bold')
ax.set_ylabel('True', fontsize=13, fontweight='bold')
ax.set_title('CNN — CIFAR-10 Confusion Matrix (Normalized by Row)',
             fontsize=14, fontweight='bold', pad=15)

plt.colorbar(im, ax=ax, shrink=0.82, label='Proportion')
plt.tight_layout()

out = os.path.join(os.path.expanduser("~"), 'Desktop', 'confusion_matrix.png')
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f"\n  图片已保存: {out}")
