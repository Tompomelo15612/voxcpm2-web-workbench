"""
VoxCPM2 模型下载脚本 (ModelScope 魔搭)
多线程断点续传，国内速度快
"""
from pathlib import Path

from modelscope import snapshot_download

MODEL_ID = "OpenBMB/VoxCPM2"
BASE_DIR = Path(__file__).resolve().parent
SAVE_DIR = BASE_DIR / "models" / "VoxCPM2"

print(f"正在从 ModelScope 下载: {MODEL_ID}")
print(f"保存路径: {SAVE_DIR}")
print("约 5GB，请耐心等待...\n")

snapshot_download(MODEL_ID, local_dir=str(SAVE_DIR))

print(f"\n[OK] 下载完成！运行: python {BASE_DIR / 'test.py'}")
