# VoxCPM2 环境搭建与使用指南

> 项目：OpenBMB/VoxCPM2 — 无分词器多语言 TTS 语音合成
> 环境：Anaconda + Windows/Linux + NVIDIA GPU
> 日期：2026-06-02

---

## 0. 新机一键搭建（全部命令，逐条复制执行）

```bash
# ===== 第1步：创建虚拟环境 =====
conda create -n voice python=3.12 -y
conda activate voice

# ===== 第2步：配置 pip 国内镜像（可选，国内网络必做） =====
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# ===== 第3步：安装 CUDA 版 PyTorch =====
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126

# ===== 第4步：安装项目依赖 =====
pip install voxcpm torchcodec modelscope librosa soundfile numpy requests tqdm flask

# ===== 第5步：验证 CUDA 是否可用 =====
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# ===== 第6步：下载模型（二选一） =====
# 方式A：ModelScope（国内推荐）
python download_model.py

# 方式B：HuggingFace 镜像
# Linux/macOS
export HF_ENDPOINT="https://hf-mirror.com"
# Windows PowerShell
# $env:HF_ENDPOINT="https://hf-mirror.com"
# 然后首次运行代码时自动下载

# ===== 完成！运行测试 =====
python test.py
```

> **逐条执行**，等每条跑完再跑下一条。全程约 10~15 分钟（主要耗时在第6步下载模型 5GB）。

---

## 1. 硬件需求

| 项目 | 建议配置 |
|------|---------|
| GPU 显存 | 至少 8 GB |
| 内存 | 至少 16 GB |
| 磁盘 | 至少 10 GB（模型约 5 GB + 依赖） |
| CUDA | 与所安装 PyTorch 版本兼容 |

---

## 2. 创建虚拟环境

```bash
# 创建环境（Python 3.10~3.12，不支持 3.13）
conda create -n voice python=3.12 -y

# 激活环境
conda activate voice
```

> ⚠️ 不要用 Python 3.13，`pynini` / `WeTextProcessing` 等依赖没有 3.13 的 wheel。

---

## 3. 配置 pip 国内镜像（可选但强烈推荐）

```bash
# 永久设为清华源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
```

---

## 4. 安装核心依赖

```bash
# 1) CUDA 版 PyTorch（cu126）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126

# 2) VoxCPM
pip install voxcpm

# 3) 附加依赖
pip install torchcodec modelscope librosa soundfile

# 4) 验证 CUDA 可用
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

---

## 5. 下载模型

### 方式一：ModelScope 下载（国内推荐，快）

```bash
python download_model.py
```

### 方式二：Hugging Face 下载（需设镜像）

```bash
# Linux/macOS
export HF_ENDPOINT="https://hf-mirror.com"

# Windows PowerShell
$env:HF_ENDPOINT="https://hf-mirror.com"

# 代码中首次调用会自动下载（约 5 GB）
python -c "from voxcpm import VoxCPM; VoxCPM.from_pretrained('openbmb/VoxCPM2')"
```

---

## 6. 安装 FFmpeg（音频处理必需）

Windows 可从 https://ffmpeg.org/download.html 下载并将 `bin` 目录加入 PATH；Linux 可使用系统包管理器安装，例如 Ubuntu/Debian 执行 `apt install ffmpeg`。

验证：
```bash
ffmpeg -version
```

---

## 7. 最小可用脚本

```python
from pathlib import Path
from voxcpm import VoxCPM
import soundfile as sf

model_dir = Path(__file__).resolve().parent / "models" / "VoxCPM2"
model = VoxCPM.from_pretrained(str(model_dir), load_denoiser=False)

wav = model.generate(
    text="你好，欢迎使用VoxCPM2语音合成。",
    cfg_value=2.0,
    inference_timesteps=10,
)
sf.write("output.wav", wav, model.tts_model.sample_rate)
```

---

## 8. 核心功能用法

### 8.1 配音设计（无参考音频，纯文字捏声音）

格式：`(声音描述)正文内容`

```python
wav = model.generate(
    text="(一位成熟女性，声音冷静低沉，语速偏慢)是，我是沈若。",
    cfg_value=2.0,
    inference_timesteps=10,
)
```

可描述维度：性别、年龄、语速、情绪、口音、场景。

### 8.2 声音克隆（方法一：仅参考音频）

```python
wav = model.generate(
    text="要合成的内容",
    reference_wav_path="参考音频.wav",   # 只锁音色
)
```

### 8.3 极致克隆（方法二：参考音频 + 对应文字稿）

```python
wav = model.generate(
    text="要合成的内容",
    prompt_wav_path="参考音频.mp3",       # 参考音频
    prompt_text="参考音频对应的文字稿",     # 文字稿
    reference_wav_path="参考音频.mp3",     # 进一步提升相似度
)
```

> ⚠️ 克隆模式下，text 中**不要加括号描述**，会被当成正文读出来。
> ⚠️ 多次 `generate()` 需传同一个 `reference_wav_path` 才能保持同一个声音。

### 8.4 语速调整（后处理）

VoxCPM2 无内置语速参数，用 `librosa` 后处理：

```python
import librosa
import numpy as np

SPEED = 0.85  # <1 变慢，>1 变快

wav = model.generate(text="...")
wav_stretched = librosa.effects.time_stretch(
    y=wav.astype(np.float32), rate=SPEED
)
```

| SPEED | 效果 |
|-------|------|
| 0.75 | 慢 25% |
| 0.85 | 慢 15%（推荐"冷静克制"风格） |
| 1.00 | 原速 |
| 1.15 | 快 15% |
| 1.25 | 快 25% |

### 8.5 多段拼接（情绪递进）

当一段台词情绪变化剧烈时，拆段生成后拼接：

```python
import numpy as np

segments = [
    "第一段平静的话…",
    "第二段逐渐激动…",
    "第三段爆发！",
]

all_wavs = []
for seg in segments:
    wav = model.generate(
        text=seg,
        reference_wav_path="参考音频.mp3",  # 每段传同一个参考，锁住声音
        cfg_value=2.0,
        inference_timesteps=10,
    )
    all_wavs.append(wav)

final = np.concatenate(all_wavs)
sf.write("output.wav", final, model.tts_model.sample_rate)
```

---

## 9. 参数速查

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `text` | str | 必填 | 要合成的文本 |
| `prompt_wav_path` | str | None | 提示音频路径（用于极致克隆） |
| `prompt_text` | str | None | 提示音频对应文字稿 |
| `reference_wav_path` | str | None | 参考音频路径（克隆音色） |
| `cfg_value` | float | 2.0 | 引导强度，越高越稳定但可能生硬 |
| `inference_timesteps` | int | 10 | 推理步数，越大质量越高但越慢 |
| `min_len` | int | 2 | 最短音频长度 |
| `max_len` | int | 4096 | 最大 token 长度 |
| `normalize` | bool | False | 文本正则化 |
| `denoise` | bool | False | 参考音频降噪 |

---

## 10. 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| pip SSL 错误 | 国内网络干扰 | 设清华镜像 (`pip config set`) |
| 模型文件缺失 | HF 下载不完整（LFS 大文件没拉） | 用 ModelScope 重下 |
| 多人声串话 | 多次 `generate()` 无参考音频，每次捏不同声音 | 传同一个 `reference_wav_path` |
| 括号描述被读出来 | 克隆模式下不支持括号描述 | 去掉括号描述，靠标点控制语气 |
| Python 3.13 装不上 | 依赖包无 3.13 wheel | 用 Python 3.12 |
| CUDA 不可用 | PyTorch 装的 CPU 版 | `pip install torch --index-url https://download.pytorch.org/whl/cu126` |

---

## 11. 目录结构

```
voxcpm2-web-workbench/
├── SETUP.md                  ← 本文件
├── test.py                   ← 无需参考音频的最小测试脚本
├── download_model.py         ← 模型下载脚本
├── models/
│   └── VoxCPM2/              ← 本地模型文件（5 GB）
├── references/               ← 用户上传的参考音频
└── outputs/                  ← 生成的音频文件
```
