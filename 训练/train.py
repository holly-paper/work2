import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
import torchvision, torchvision.transforms as T
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import numpy as np, os, sys, time

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== 工具函数 =====================

def fmt_time(sec):
    """格式化时间显示"""
    if sec < 60:       return f"{sec:.0f}s"
    if sec < 3600:     return f"{int(sec)//60}m{int(sec)%60:02d}s"
    h, m = int(sec)//3600, (int(sec)%3600)//60
    return f"{h}h{m:02d}m" 

def progress_bar(pct, width=30):
    """ASCII 进度条"""
    filled = int(width * pct / 100)
    return "[" + "=" * filled + ">" + " " * (width - filled - 1) + "]"

# ===================== 1. 数据加载 =====================
print("=" * 60)
print("  CNN vs MLP — CIFAR-10 图像分类实验")
print(f"  设备: {DEV.__str__().upper()}")
print("=" * 60)

print("\n[1/5] 加载 CIFAR-10 数据集...")
sys.stdout.flush()

tf_train = T.Compose([
    T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
    T.ToTensor(), T.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616)),
])
tf_test = T.Compose([
    T.ToTensor(), T.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616)),
])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True,  download=True, transform=tf_train)
test_set  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=tf_test)

N = len(train_set)
val_set, train_set = torch.utils.data.random_split(
    train_set, [5000, N-5000], generator=torch.Generator().manual_seed(42))

tr_loader = DataLoader(train_set, batch_size=64, shuffle=True,  num_workers=0)
va_loader = DataLoader(val_set,   batch_size=64, shuffle=False, num_workers=0)
te_loader = DataLoader(test_set,   batch_size=64, shuffle=False, num_workers=0)

print(f"  训练: {len(train_set):,}  验证: {len(val_set):,}  测试: {len(test_set):,}")
print(f"  类别: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck")
sys.stdout.flush()

# ===================== 2. 模型 =====================

print("\n[2/5] 构建模型...")

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.Conv2d(32,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.Conv2d(64,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.Conv2d(128,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(128*4*4,256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256,10),
        )
    def forward(self,x): return self.net(x)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3072,1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024,512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512,10),
        )
    def forward(self,x): return self.net(x)

cnn_params = sum(p.numel() for p in CNN().parameters())
mlp_params = sum(p.numel() for p in MLP().parameters())
print(f"  CNN 参数: {cnn_params:,}    MLP 参数: {mlp_params:,}   (CNN = MLP 的 {cnn_params/mlp_params*100:.1f}%)")
sys.stdout.flush()

# ===================== 3. 训练引擎 =====================

@torch.no_grad()
def evaluate(model, loader, crit):
    model.eval(); loss_sum, ok, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEV), y.to(DEV)
        out = model(x)
        loss_sum += crit(out, y).item() * x.size(0)
        ok += (out.argmax(1) == y).sum().item(); n += x.size(0)
    return loss_sum / n, 100. * ok / n

def train_model(model, name, epochs=50):
    print(f"\n{'='*60}")
    print(f"  [3/5]  训练 {name}")
    print(f"  {'='*60}")
    print(f"  Epoch | Train Acc | Valid Acc | Loss    | 耗时   | 预计剩余")
    print(f"  {'-'*60}")
    sys.stdout.flush()

    model = model.to(DEV)
    crit = nn.CrossEntropyLoss()
    opt  = optim.Adam(model.parameters(), lr=1e-3)
    sched = optim.lr_scheduler.StepLR(opt, 20, 0.5)

    hist = {'tr_acc': [], 'va_acc': [], 'tr_loss': [], 'va_loss': []}
    t_start = time.time()

    for ep in range(1, epochs + 1):
        # ---- 训练一个 epoch ----
        model.train()
        for x, y in tr_loader:
            x, y = x.to(DEV), y.to(DEV); opt.zero_grad()
            crit(model(x), y).backward(); opt.step()
        sched.step()

        # ---- 评估 ----
        tr_loss, tr_acc = evaluate(model, tr_loader, crit)
        va_loss, va_acc = evaluate(model, va_loader, crit)
        hist['tr_acc'].append(tr_acc);  hist['va_acc'].append(va_acc)
        hist['tr_loss'].append(tr_loss); hist['va_loss'].append(va_loss)

        # ---- 进度显示 ----
        elapsed    = time.time() - t_start
        eta        = (elapsed / ep) * (epochs - ep) if ep > 0 else 0
        bar        = progress_bar(ep * 100 / epochs, 20)

        print(f"  {bar} {ep:3d}/{epochs} | "
              f"{tr_acc:5.1f}%   | {va_acc:5.1f}%   | "
              f"{tr_loss:.4f}  | {fmt_time(elapsed):>6s} | {fmt_time(eta):>6s}")
        sys.stdout.flush()

    # 最终测试
    _, te_acc = evaluate(model, te_loader, crit)
    total_time = time.time() - t_start
    print(f"  {'-'*60}")
    print(f"  {name} 完成!  测试准确率: {te_acc:.2f}%   总耗时: {fmt_time(total_time)}")
    sys.stdout.flush()
    return hist, te_acc

# ===================== 4. 训练 =====================

cnn_hist, cnn_test = train_model(CNN(), "CNN", 50)
mlp_hist, mlp_test = train_model(MLP(), "MLP", 50)

# ===================== 5. 绘图 =====================

print(f"\n[4/5] 绘制对比曲线...")
sys.stdout.flush()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, hist, name, c in [
    (axes[0], cnn_hist, 'CNN', '#1f77b4'),
    (axes[1], mlp_hist, 'MLP', '#d62728'),
]:
    eps = range(1, 51)
    ax.plot(eps, hist['tr_acc'], '-',  color=c, lw=1.5, label='Training Accuracy')
    ax.plot(eps, hist['va_acc'], '--', color=c, lw=1.5, label='Validation Accuracy')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'{name}  (Best Val: {max(hist["va_acc"]):.1f}%)', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10); ax.grid(alpha=0.3); ax.set_ylim(0, 100)

fig.suptitle('CNN vs MLP — CIFAR-10 Accuracy Curves', fontsize=15, fontweight='bold')
plt.tight_layout()

out = os.path.join(os.path.expanduser("~"), 'Desktop', 'cnn_vs_mlp_accuracy.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"  图片已保存: {out}")
sys.stdout.flush()

# ===================== 6. 汇总 =====================

print(f"\n[5/5] {'='*50}")
print(f"  实验结果汇总")
print(f"  {'='*50}")
print(f"  CNN  测试准确率 : {cnn_test:6.2f}%   参数: {cnn_params:,}")
print(f"  MLP  测试准确率 : {mlp_test:6.2f}%   参数: {mlp_params:,}")
print(f"  CNN 提升        : {cnn_test - mlp_test:6.1f} 个百分点")
print(f"  CNN 收敛 epoch1 : {cnn_hist['tr_acc'][0]:.1f}%  → epoch10: {cnn_hist['tr_acc'][9]:.1f}%")
print(f"  MLP 收敛 epoch1 : {mlp_hist['tr_acc'][0]:.1f}%  → epoch10: {mlp_hist['tr_acc'][9]:.1f}%")
print(f"  {'='*50}")
