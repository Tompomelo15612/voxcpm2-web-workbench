from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from voxcpm import VoxCPM

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "VoxCPM2"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

model = VoxCPM.from_pretrained(
    str(MODEL_DIR),
    load_denoiser=False,
)

TEXT = "(温暖、清晰、自然的中文旁白)你好，VoxCPM2 语音生成工作台已经成功运行。"

# 使用声音设计模式进行最小测试，无需准备参考音频。
wav = model.generate(
    text=TEXT,
    cfg_value=2.0,
    inference_timesteps=10,
)
sample_rate = model.tts_model.sample_rate

SPEED = 0.85  # <1.0 变慢，>1.0 变快（例：0.85 = 慢15%，1.2 = 快20%）

wav_adjusted = librosa.effects.time_stretch(y=wav.astype(np.float32), rate=SPEED)
speed_output = OUTPUT_DIR / f"smoke_test_speed{SPEED}.wav"
original_output = OUTPUT_DIR / "smoke_test_original.wav"
sf.write(str(speed_output), wav_adjusted, sample_rate)

sf.write(str(original_output), wav, sample_rate)

print(f"[OK] {original_output} ({len(wav)/sample_rate:.1f}s)")
print(f"[OK] {speed_output} ({len(wav_adjusted)/sample_rate:.1f}s) - 速度 x{SPEED}")
print("\n调速说明：修改脚本里 SPEED 值即可")
print(f"  0.80 = 慢20% | 0.90 = 慢10% | 1.0 = 原速 | 1.10 = 快10% | 1.20 = 快20%")
