# work2
# CNN vs MLP — CIFAR-10 图像分类对比实验

> **《人工智能》课程论文项目**  
> 题目：基于卷积神经网络的图像分类算法研究  
> 框架：PyTorch 2.x  |  语言：Python 3.10+  |  平台：Windows / Linux / macOS

---

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 文件结构](#2-文件结构)
- [3. 环境配置](#3-环境配置)
- [4. 数据集](#4-数据集)
- [5. 运行步骤](#5-运行步骤)
- [6. 模块说明](#6-模块说明)
- [7. 实验结果](#7-实验结果)
- [8. 常见问题](#8-常见问题)
- [9. 参考文献](#9-参考文献)

---

## 1. 项目概述

本项目实现了一个基于卷积神经网络（CNN）的 CIFAR-10 图像分类器，并与同等参数规模的全连接网络（MLP）进行对比实验。项目包含三个核心实验脚本和一个论文生成脚本：

| 脚本 | 功能 | 输出 | 论文位置 |
|------|------|------|---------|
| `cnn_vs_mlp_final.py` | 训练 CNN 和 MLP，绘制准确率对比曲线 | `cnn_vs_mlp_accuracy.png` | 图1 |
| `confusion_matrix_cnn.py` | 训练 CNN，生成归一化混淆矩阵 | `confusion_matrix.png` | 图2 |
| `conv_filters_cnn.py` | 训练 CNN，可视化第一层卷积核 | `conv_filters.png` | 图3 |

---

## 2. 文件结构

```
桌面/
├── README.md                          ← 本文件
│
├── cnn_vs_mlp_final.py                ← CNN vs MLP 训练 + 准确率曲线（图1）
├── confusion_matrix_cnn.py            ← CNN 混淆矩阵（图2）
├── conv_filters_cnn.py                ← CNN 卷积核可视化（图3）
│
├── cnn_vs_mlp_accuracy.png            ← 输出：图1（运行后生成）
├── confusion_matrix.png               ← 输出：图2（运行后生成）
├── conv_filters.png                   ← 输出：图3（运行后生成）
│
├── AI课程论文_CNN图像分类_定稿.docx     ← 课程论文 Word 文档
│
├── 图像复原.py                         ← DIP 图像复原 GUI（独立项目）
│
└── data/                              ← CIFAR-10 数据（自动下载）
    └── cifar-10-batches-py/
```

---

## 3. 环境配置

### 3.1 硬件要求

| 配置项 | 最低要求 | 推荐配置 |
|--------|---------|---------|
| CPU | Intel Core i5 / AMD Ryzen 5 | Intel Core i7+ |
| 内存 | 8 GB | 16 GB+ |
| GPU（可选） | NVIDIA CUDA 6GB+ VRAM | NVIDIA RTX 3060+ |
| 磁盘 | 500 MB（数据 + 模型） | 1 GB+ |

> **GPU 说明**：有 GPU 时训练速度提升 10~20 倍。无 GPU 也可运行（自动检测，使用 CPU）。

### 3.2 软件依赖

**Python 3.10 或更高版本**。安装依赖：

```bash
pip install torch torchvision matplotlib numpy pillow
```

完整依赖清单：

| 包名 | 版本要求 | 用途 |
|------|---------|------|
| `torch` | ≥ 2.0 | 深度学习框架 |
| `torchvision` | ≥ 0.15 | 数据增强、CIFAR-10 数据集（可选） |
| `matplotlib` | ≥ 3.5 | 绘制曲线、混淆矩阵、卷积核可视化 |
| `numpy` | ≥ 1.22 | 数值计算 |
| `Pillow` | ≥ 9.0 | 图像读取 |

> 若使用 NVIDIA GPU，请安装 CUDA 版 PyTorch：  
> `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

### 3.3 环境验证

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

预期输出：

```
PyTorch 2.x.x
CUDA: True    # 有 GPU
CUDA: False   # 仅 CPU
```

---

## 4. 数据集

### 4.1 方式一：自动下载（推荐）

脚本会自动从 PyTorch CDN 下载 CIFAR-10（约 170 MB），保存在 `./data/` 目录。首次运行需联网，后续运行自动跳过。

```python
# 默认使用 torchvision 内置下载
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, ...)
```

### 4.2 方式二：本地文件夹

若已下载 CIFAR-10 的 JPG 图片，修改脚本顶部的路径变量：

```python
TRAIN_DIR = r"D:\作业\train"   # 替换为你的训练集路径
TEST_DIR  = r"D:\作业\test"    # 替换为你的测试集路径
```

图片命名格式要求：`{类别标签}_{图片编号}.jpg`，例如 `0_10008.jpg` 表示类别 0（airplane）。

> 三个脚本默认使用**方式二**（本地文件夹）。若使用方式一，将 `CIFAR10Folder` 替换为 `torchvision.datasets.CIFAR10`。

---

## 5. 运行步骤

### 步骤 1：生成图1（CNN vs MLP 准确率曲线）

```bash
python cnn_vs_mlp_final.py
```

- **耗时**：CPU 约 1.5~2 小时，GPU 约 15~20 分钟
- **输出**：桌面 `cnn_vs_mlp_accuracy.png`
- **内容**：左右两个子图，分别展示 CNN 和 MLP 的 Training / Validation 准确率曲线

### 步骤 2：生成图2（CNN 混淆矩阵）

```bash
python confusion_matrix_cnn.py
```

- **耗时**：CPU 约 50 分钟，GPU 约 5~8 分钟
- **输出**：桌面 `confusion_matrix.png`
- **内容**：10×10 归一化混淆矩阵，对角线为各类召回率，右侧 colorbar

### 步骤 3：生成图3（CNN 卷积核可视化）

```bash
python conv_filters_cnn.py
```

- **耗时**：CPU 约 30 分钟，GPU 约 3~5 分钟
- **输出**：桌面 `conv_filters.png`
- **内容**：4×8 网格，每个格为 3×3 像素的 RGB 彩色卷积核

### 步骤 4（可选）：生成课程论文

论文已预生成于桌面 `AI课程论文_CNN图像分类_定稿.docx`。将三张图插入对应位置即可提交。

---

## 6. 模块说明

### 6.1 公共模块（三个脚本共用）

#### `CIFAR10Folder` — 自定义 Dataset 类

```python
class CIFAR10Folder(Dataset):
    """从本地 JPG 文件夹加载 CIFAR-10，文件名格式 {label}_{id}.jpg"""
    def __init__(self, folder, transform=None):
        # 扫描文件夹 → 解析标签 → 存储文件列表
    def __getitem__(self, idx):
        # PIL 读取 → transform → 返回 (image_tensor, label)
```

#### `CNN` / `MLP` — 模型类

```python
class CNN(nn.Module):
    """6 层卷积 + 3 层池化 + 2 层全连接
       Block1: Conv(3→32)×2 + MaxPool
       Block2: Conv(32→64)×2 + MaxPool
       Block3: Conv(64→128)×2 + MaxPool
       Classifier: Flatten → FC(2048→256) → Dropout(0.5) → FC(256→10)
    """
```

#### `fmt_time()` / `progress_bar()` — 工具函数

```python
def fmt_time(sec):
    """秒数 → 可读时间字符串，如 '1h12m03s'"""

def progress_bar(pct, width=20):
    """百分比 → ASCII 进度条，如 '[======>            ]'"""
```

### 6.2 各脚本独特模块

| 脚本 | 独特模块 | 说明 |
|------|---------|------|
| `cnn_vs_mlp_final.py` | `train_model()` | 通用训练引擎，含每 epoch 评估和进度打印 |
| | `evaluate()` | 计算验证集/测试集的 Loss 和 Accuracy |
| `confusion_matrix_cnn.py` | 混淆矩阵计算 | `np.zeros((10,10))` 统计 → 归一化 → Matplotlib 热力图 |
| `conv_filters_cnn.py` | 权重提取+可视化 | `model.conv1_1.weight` → `np.transpose` → `imshow(nearest)` |

### 6.3 数据流

```
本地 JPG 文件夹                  训练循环                  输出
┌──────────────┐    transform      ┌──────────┐    eval     ┌──────────────┐
│ D:\作业\train │ ──────────────→ │ CNN/MLP  │ ─────────→ │ 图1: 准确率   │
│   50,000 JPG  │   Dataset+Loader │  模型    │            │ 图2: 混淆矩阵 │
│ D:\作业\test  │                 │ 50 epoch │            │ 图3: 卷积核   │
│   10,000 JPG  │                 └──────────┘            └──────────────┘
└──────────────┘
```

---

## 7. 实验结果

### 7.1 预期数值

| 指标 | CNN | MLP |
|------|-----|-----|
| 参数量 | ~814,000 | ~3,677,000 |
| 测试准确率 | ~82% | ~39% |
| Epoch 1 训练准确率 | ~35% | ~28% |
| Epoch 10 训练准确率 | ~75% | ~31% |
| 训练耗时 (CPU) | ~72 分钟 | ~29 分钟 |

### 7.2 各类准确率（图2）

| 类别 | 准确率 | 类别 | 准确率 |
|------|:-----:|------|:-----:|
| truck | 91.0% | dog | 76.3% |
| automobile | 89.0% | bird | 72.6% |
| ship | 88.9% | cat | 58.3% |
| horse | 86.8% | — | — |
| airplane | 84.6% | — | — |
| deer | 80.4% | — | — |
| frog | 80.4% | — | — |

> **注意**：由于随机初始化、数据增强等因素，实际数值可能与上表有 ±3% 的浮动，属于正常范围。

---

## 8. 常见问题

### Q1：`ModuleNotFoundError: No module named 'torch'`

**A**：未安装 PyTorch。运行 `pip install torch torchvision`。

### Q2：`FileNotFoundError: D:\作业\train`

**A**：本地数据路径不存在。改为使用 torchvision 自动下载：

```python
# 将脚本中的 CIFAR10Folder 替换为：
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=tf_train)
testset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=tf_test)
```

### Q3：下载 CIFAR-10 太慢

**A**：使用镜像源：

```bash
pip install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple
```

或在脚本中设置环境变量：

```python
os.environ['TORCH_HOME'] = './torch_cache'
```

### Q4：训练速度太慢

**A**：
1. 将 `epochs` 减到 30 → 仍能得到有意义的结果
2. 减小 `batch_size` 到 32（内存不足时）
3. 安装 CUDA 版 PyTorch 使用 GPU

### Q5：图片中文显示为方块

**A**：检查系统是否安装中文字体：

```python
import matplotlib.font_manager as fm
print([f.name for f in fm.fontManager.ttflist if 'YaHei' in f.name or 'SimHei' in f.name])
```

若无，安装 `SimHei` 或修改 `matplotlib.rcParams['font.sans-serif']` 为系统已有中文字体。

### Q6：三张图的数据不完全一致

**A**：三个脚本各自独立训练模型，因随机性导致数值略有浮动。如需完全一致，可使用同一个训练好的模型（保存 `.pth` 文件并在其他脚本中加载）。

---



---

> 📧 如有问题，请联系作者。  
> 📅 最后更新：2026 年 6 月

