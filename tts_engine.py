from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "VoxCPM2"
REFERENCES_DIR = BASE_DIR / "references"
OUTPUTS_DIR = BASE_DIR / "outputs"

REFERENCE_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
OUTPUT_EXTENSIONS = {".wav"}
GENERATION_MODES = {"clone", "ultimate", "voice_design"}
MAX_TEXT_CHARS = 10_000

_model = None
_model_lock = threading.Lock()
_generation_lock = threading.Lock()


@dataclass(frozen=True)
class ReferenceAudio:
    key: str
    name: str
    path: Path
    location: str
    duration: float | None = None
    sample_rate: int | None = None


@dataclass(frozen=True)
class AudioItem:
    key: str
    name: str
    path: Path
    size_mb: float
    modified: datetime
    duration: float | None
    sample_rate: int | None


@dataclass(frozen=True)
class GenerationResult:
    filename: str
    path: Path
    duration: float | None
    sample_rate: int
    speed: float
    tone_label: str
    language_key: str
    segment_count: int


@dataclass(frozen=True)
class TonePreset:
    key: str
    label: str
    description: str
    voice_design_prompt: str
    suggested_speed: float


class BusyError(RuntimeError):
    pass


TONE_PRESETS: dict[str, TonePreset] = {
    "natural": TonePreset(
        key="natural",
        label="自然",
        description="保留原文本节奏，不额外处理。",
        voice_design_prompt="natural speaking voice, clear and steady",
        suggested_speed=1.0,
    ),
    "calm": TonePreset(
        key="calm",
        label="冷静克制",
        description="增加句间停顿，适合压迫感、复仇独白、低情绪表达。",
        voice_design_prompt="calm, restrained, low voice, slow pace, controlled emotion",
        suggested_speed=0.85,
    ),
    "dramatic": TonePreset(
        key="dramatic",
        label="剧情独白",
        description="强化段落感，适合短剧旁白和情绪递进。",
        voice_design_prompt="dramatic narration, cinematic, emotional but controlled",
        suggested_speed=0.9,
    ),
    "angry": TonePreset(
        key="angry",
        label="愤怒爆发",
        description="保留强标点并拉开短句，适合质问、爆发、冲突台词。",
        voice_design_prompt="angry, intense, powerful, fast emotional delivery",
        suggested_speed=1.05,
    ),
    "gentle": TonePreset(
        key="gentle",
        label="温柔旁白",
        description="放慢句尾节奏，适合温柔、安抚、柔和叙述。",
        voice_design_prompt="gentle, warm, soft voice, soothing and tender",
        suggested_speed=0.9,
    ),
    "suspense": TonePreset(
        key="suspense",
        label="悬疑压低",
        description="制造更明显的停顿，适合悬疑、反转、压低声线。",
        voice_design_prompt="low suspenseful voice, mysterious, slow, tense pauses",
        suggested_speed=0.85,
    ),
}


def ensure_dirs() -> None:
    REFERENCES_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)


def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from voxcpm import VoxCPM

                _model = VoxCPM.from_pretrained(
                    str(MODEL_DIR),
                    load_denoiser=False,
                )
    return _model


def _audio_duration(path: Path) -> tuple[float | None, int | None]:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return info.frames / info.samplerate, info.samplerate
    except Exception:
        return None, None


def _iter_audio_files(paths: Iterable[Path], extensions: set[str]) -> Iterable[Path]:
    seen: set[Path] = set()
    for directory in paths:
        if not directory.exists():
            continue
        for path in directory.iterdir():
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield path


def list_references() -> list[ReferenceAudio]:
    ensure_dirs()
    items: list[ReferenceAudio] = []
    for path in _iter_audio_files([REFERENCES_DIR, BASE_DIR], REFERENCE_EXTENSIONS):
        location = "references" if path.parent.resolve() == REFERENCES_DIR.resolve() else "root"
        if location == "root" and path.name.lower().startswith("demo_"):
            continue
        duration, sample_rate = _audio_duration(path)
        items.append(
            ReferenceAudio(
                key=f"{location}:{path.name}",
                name=path.name,
                path=path,
                location=location,
                duration=duration,
                sample_rate=sample_rate,
            )
        )
    return sorted(items, key=lambda item: (item.location != "references", item.name.lower()))


def list_outputs() -> list[AudioItem]:
    ensure_dirs()
    items: list[AudioItem] = []
    for path in _iter_audio_files([OUTPUTS_DIR, BASE_DIR], OUTPUT_EXTENSIONS):
        if path.parent.resolve() == BASE_DIR.resolve() and not path.name.lower().startswith("demo_"):
            continue
        duration, sample_rate = _audio_duration(path)
        location = "outputs" if path.parent.resolve() == OUTPUTS_DIR.resolve() else "root"
        stat = path.stat()
        items.append(
            AudioItem(
                key=f"{location}:{path.name}",
                name=path.name,
                path=path,
                size_mb=stat.st_size / 1024 / 1024,
                modified=datetime.fromtimestamp(stat.st_mtime),
                duration=duration,
                sample_rate=sample_rate,
            )
        )
    return sorted(items, key=lambda item: item.modified, reverse=True)


def list_tone_presets() -> list[TonePreset]:
    return list(TONE_PRESETS.values())


def get_tone_preset(key: str | None) -> TonePreset:
    return TONE_PRESETS.get(key or "natural", TONE_PRESETS["natural"])


def resolve_reference(key: str) -> Path:
    for item in list_references():
        if item.key == key:
            return item.path
    raise FileNotFoundError("参考音频不存在")


def resolve_audio(key: str) -> Path:
    location, _, filename = key.partition(":")
    if not filename:
        raise FileNotFoundError("音频文件不存在")

    directory = OUTPUTS_DIR if location == "outputs" else BASE_DIR if location == "root" else None
    if directory is None:
        raise FileNotFoundError("音频文件不存在")

    path = (directory / filename).resolve()
    if directory.resolve() not in [path, *path.parents]:
        raise FileNotFoundError("音频文件不存在")
    if not path.exists() or path.suffix.lower() not in OUTPUT_EXTENSIONS:
        raise FileNotFoundError("音频文件不存在")
    return path


def clean_filename(value: str | None, fallback: str = "tts_output") -> str:
    stem = (value or "").strip()
    if not stem:
        stem = fallback
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", stem)
    stem = re.sub(r"\s+", "_", stem).strip("._ ")
    if not stem:
        stem = fallback
    return stem[:80]


def unique_output_path(output_name: str | None) -> Path:
    ensure_dirs()
    stem = clean_filename(output_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = OUTPUTS_DIR / f"{stem}_{timestamp}.wav"
    index = 2
    while candidate.exists():
        candidate = OUTPUTS_DIR / f"{stem}_{timestamp}_{index}.wav"
        index += 1
    return candidate


def save_uploaded_reference(filename: str, stream) -> ReferenceAudio:
    ensure_dirs()
    suffix = Path(filename).suffix.lower()
    if suffix not in REFERENCE_EXTENSIONS:
        raise ValueError("只支持 wav、mp3、flac、m4a、ogg 参考音频")

    stem = clean_filename(Path(filename).stem, "reference")
    target = REFERENCES_DIR / f"{stem}{suffix}"
    index = 2
    while target.exists():
        target = REFERENCES_DIR / f"{stem}_{index}{suffix}"
        index += 1
    stream.save(target)
    duration, sample_rate = _audio_duration(target)
    return ReferenceAudio(
        key=f"references:{target.name}",
        name=target.name,
        path=target,
        location="references",
        duration=duration,
        sample_rate=sample_rate,
    )


def _compact_blank_lines(text: str) -> str:
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _line_break_sentences(text: str, *, paragraph_break: bool = False) -> str:
    replacement = r"\1\n\n" if paragraph_break else r"\1\n"
    text = re.sub(r"([。！？!?；;])\s*", replacement, text)
    return _compact_blank_lines(text)


def _leading_instruction(text: str) -> tuple[str | None, str]:
    match = re.match(r"^\s*([（(][^）)]{1,180}[）)])\s*(.*)$", text, flags=re.S)
    if not match:
        return None, text
    return match.group(1), match.group(2).strip()


def _with_instruction(text: str, instruction: str | None) -> str:
    if not instruction:
        return text
    if text.startswith("(") or text.startswith("（"):
        return text
    return f"{instruction}{text}"


def _audio_peak(wav) -> float:
    try:
        import numpy as np

        if wav is None:
            return 0.0
        array = np.asarray(wav)
        if array.size == 0:
            return 0.0
        array = np.nan_to_num(array.astype("float32"), nan=0.0, posinf=0.0, neginf=0.0)
        return float(np.max(np.abs(array)))
    except Exception:
        if wav is None:
            return 0.0
        try:
            values = list(wav)
        except TypeError:
            return 0.0
        if not values:
            return 0.0
        return max(abs(float(value)) for value in values if value == value)


def _is_silent(wav) -> bool:
    return _audio_peak(wav) < 1e-5


def _time_stretch(wav, speed: float):
    import librosa
    import numpy as np

    return librosa.effects.time_stretch(y=wav.astype(np.float32), rate=speed)


def _slow_down_with_pauses(wav, speed: float, sample_rate: int):
    import librosa
    import numpy as np

    if speed >= 1.0:
        return _time_stretch(wav, speed)

    source = np.asarray(wav, dtype=np.float32)
    if source.size == 0:
        return source

    target_samples = int(round(len(source) / speed))
    remaining = target_samples - len(source)
    if remaining <= int(sample_rate * 0.02):
        return source

    intervals = librosa.effects.split(
        source,
        top_db=32,
        frame_length=2048,
        hop_length=512,
    )
    if len(intervals) < 2:
        return source

    insertions = len(intervals) - 1
    base_pause = remaining // insertions
    extra = remaining % insertions
    pieces = []
    cursor = 0
    for index, (_, end) in enumerate(intervals[:-1]):
        pieces.append(source[cursor:end])
        pause_samples = base_pause + (1 if index < extra else 0)
        if pause_samples > 0:
            pieces.append(np.zeros(pause_samples, dtype=np.float32))
        cursor = end
    pieces.append(source[cursor:])
    return np.concatenate(pieces)


def _safe_generation_text(text: str) -> str:
    _, text = _leading_instruction(text)
    text = re.sub(r"\[[^\]]{1,80}\]", "", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def apply_control_instruction(text: str, enabled: bool, instruction: str | None) -> str:
    instruction = (instruction or "").strip()
    if not enabled or not instruction:
        return text
    if not (instruction.startswith("(") or instruction.startswith("（")):
        instruction = f"({instruction})"
    return _with_instruction(text, instruction)

def apply_slow_pacing(text: str, speed: float) -> str:
    if speed >= 1.0:
        return text

    instruction, body = _leading_instruction(text)
    line_break = "\n\n" if speed <= 0.85 else "\n"
    body = re.sub(r"([。！？!?；;])\s*", r"\1" + line_break, body)
    if speed <= 0.9:
        body = re.sub(r"([，、,])\s*", r"\1\n", body)
    if speed <= 0.75:
        body = re.sub(r"([：:])\s*", r"\1\n", body)
    body = _compact_blank_lines(body)
    return _with_instruction(body, instruction)


def split_text_segments(text: str, max_chars: int) -> list[str]:
    max_chars = max(80, min(max_chars, 1200))
    text = _compact_blank_lines(text)
    if len(text) <= max_chars:
        return [text]

    instruction, body = _leading_instruction(text)
    raw_parts = re.split(r"(\n{2,}|[。！？!?；;]\s*)", body)
    units: list[str] = []
    for index in range(0, len(raw_parts), 2):
        unit = raw_parts[index]
        if index + 1 < len(raw_parts):
            unit += raw_parts[index + 1]
        unit = unit.strip()
        if unit:
            units.append(unit)

    segments: list[str] = []
    current = ""
    for unit in units:
        if len(unit) > max_chars:
            comma_parts = [part for part in re.split(r"(?<=[，,、])", unit) if part.strip()]
        else:
            comma_parts = [unit]

        for part in comma_parts:
            part = part.strip()
            if not part:
                continue
            if current and len(current) + len(part) + 1 > max_chars:
                segments.append(current.strip())
                current = part
            else:
                current = f"{current}\n{part}" if current else part

    if current:
        segments.append(current.strip())
    if not segments:
        segments = [body]

    return [_with_instruction(segment, instruction) for segment in segments]


def apply_tone(text: str, mode: str, tone_key: str | None) -> tuple[str, TonePreset]:
    preset = get_tone_preset(tone_key)
    text = _compact_blank_lines(text)
    if preset.key == "natural":
        return text, preset

    if mode == "voice_design":
        if text.startswith("(") or text.startswith("（"):
            return text, preset
        return f"({preset.voice_design_prompt}){text}", preset

    if preset.key == "calm":
        text = _line_break_sentences(text)
    elif preset.key == "dramatic":
        text = _line_break_sentences(text, paragraph_break=True)
    elif preset.key == "angry":
        text = _line_break_sentences(text)
        text = re.sub(r"。\s*$", "！", text)
    elif preset.key == "gentle":
        text = _line_break_sentences(text)
    elif preset.key == "suspense":
        text = _line_break_sentences(text)

    return _compact_blank_lines(text), preset


def synthesize(
    *,
    text: str,
    mode: str,
    language_key: str,
    reference_key: str | None,
    prompt_text: str | None,
    tone_key: str | None,
    cfg_value: float,
    inference_timesteps: int,
    speed: float,
    normalize: bool,
    denoise: bool,
    retry_badcase: bool,
    auto_split: bool,
    max_segment_chars: int,
    enable_control: bool,
    control_instruction: str | None,
    output_name: str | None,
    progress_callback: Callable[[int, str], None] | None = None,
) -> GenerationResult:
    def report(percent: int, message: str) -> None:
        if progress_callback:
            progress_callback(percent, message)

    text = text.strip()
    if not text:
        raise ValueError("请输入要合成的文本")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"文本不能超过 {MAX_TEXT_CHARS} 个字符")
    if mode not in GENERATION_MODES:
        raise ValueError("不支持的生成模式")
    if not 0 <= cfg_value <= 10:
        raise ValueError("CFG 必须在 0 到 10 之间")
    if not 1 <= inference_timesteps <= 100:
        raise ValueError("推理步数必须在 1 到 100 之间")
    if not 0.5 <= speed <= 2.0:
        raise ValueError("语速必须在 0.5 到 2.0 之间")

    if not _generation_lock.acquire(blocking=False):
        raise BusyError("当前已有生成任务在运行，请稍后再试")
    try:
        report(5, "整理文本和参数")
        text, tone = apply_tone(text, mode, tone_key)
        if mode == "ultimate" and enable_control and control_instruction:
            raise ValueError("极致克隆模式会忽略控制提示词，请关闭可控克隆提示词或改用基础克隆")
        text = apply_control_instruction(text, enable_control, control_instruction)
        text = apply_slow_pacing(text, speed)
        segments = split_text_segments(text, max_segment_chars) if auto_split else [text]
        report(10, f"准备生成，共 {len(segments)} 段")
        model = get_model()
        report(18, "模型已就绪")
        kwargs = {
            "cfg_value": cfg_value,
            "inference_timesteps": inference_timesteps,
            "normalize": normalize,
            "denoise": denoise,
            "retry_badcase": retry_badcase,
        }

        if mode == "voice_design":
            pass
        elif mode == "ultimate":
            if not reference_key:
                raise ValueError("极致克隆需要选择参考音频")
            if not prompt_text or not prompt_text.strip():
                raise ValueError("极致克隆需要填写参考音频对应文本")
            reference_path = resolve_reference(reference_key)
            kwargs["reference_wav_path"] = str(reference_path)
            kwargs["prompt_wav_path"] = str(reference_path)
            kwargs["prompt_text"] = prompt_text.strip()
        else:
            if not reference_key:
                raise ValueError("基础克隆需要选择参考音频")
            kwargs["reference_wav_path"] = str(resolve_reference(reference_key))

        wav_chunks = []
        total = len(segments)
        for index, segment in enumerate(segments, start=1):
            start_percent = 18 + int((index - 1) / total * 62)
            report(start_percent, f"正在生成第 {index}/{total} 段")
            wav_chunk = model.generate(text=segment, **kwargs)
            if _is_silent(wav_chunk):
                report(start_percent, f"第 {index}/{total} 段静音，正在安全重试")
                safe_text = _safe_generation_text(segment)
                if safe_text and safe_text != segment:
                    wav_chunk = model.generate(text=safe_text, **kwargs)
            if _is_silent(wav_chunk):
                raise ValueError(f"第 {index}/{total} 段生成静音，请换自然语气或减少情绪标签后重试")
            wav_chunks.append(wav_chunk)
            done_percent = 18 + int(index / total * 62)
            report(done_percent, f"第 {index}/{total} 段完成")
        sample_rate = model.tts_model.sample_rate

        if len(wav_chunks) == 1:
            wav = wav_chunks[0]
        else:
            import numpy as np

            report(82, "正在拼接分段音频")
            wav = np.concatenate(wav_chunks)

        if abs(speed - 1.0) > 0.001:
            report(88, "正在调整语速")
            original_wav = wav
            wav = _slow_down_with_pauses(wav, speed, sample_rate)
            if _is_silent(wav) and not _is_silent(original_wav):
                report(90, "调速结果异常，已保留原速音频")
                wav = original_wav

        import soundfile as sf

        if _is_silent(wav):
            raise ValueError("生成结果为静音，已阻止保存。请换自然语气、0.9 或减少情绪标签后重试")
        report(94, "正在保存音频")
        output_path = unique_output_path(output_name)
        sf.write(str(output_path), wav, sample_rate)
        duration, _ = _audio_duration(output_path)
        report(100, "生成完成")

        return GenerationResult(
            filename=output_path.name,
            path=output_path,
            duration=duration,
            sample_rate=sample_rate,
            speed=speed,
            tone_label=tone.label,
            language_key=language_key,
            segment_count=len(segments),
        )
    finally:
        _generation_lock.release()
