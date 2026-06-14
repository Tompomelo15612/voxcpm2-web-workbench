from __future__ import annotations

import threading
import uuid
from pathlib import Path

from flask import Flask, abort, jsonify, redirect, render_template, request, send_file, url_for

from tts_engine import (
    BusyError,
    list_outputs,
    list_references,
    list_tone_presets,
    resolve_audio,
    save_uploaded_reference,
    synthesize,
)


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024
app.config["TEMPLATES_AUTO_RELOAD"] = True

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

LANGUAGE_GROUPS = [
    (
        "自动",
        [
            ("auto", "自动识别", "输入要生成的台词或旁白"),
        ],
    ),
    (
        "常用语言",
        [
            ("zh", "中文", "请输入中文文本，例如：今晚，我会把真相说清楚。"),
            ("en", "英语", "Enter English text, for example: Tonight, I will tell the truth."),
            ("ja", "日语", "日本語の文章を入力してください。例：今夜、真実を話します。"),
            ("ko", "韩语", "한국어 문장을 입력하세요. 예: 오늘 밤, 진실을 말하겠습니다."),
            ("fr", "法语", "Entrez un texte en français. Exemple : Ce soir, je dirai la vérité."),
            ("de", "德语", "Geben Sie deutschen Text ein. Beispiel: Heute Abend sage ich die Wahrheit."),
            ("es", "西班牙语", "Escribe texto en español. Ejemplo: Esta noche diré la verdad."),
            ("ru", "俄语", "Введите русский текст. Например: Сегодня вечером я скажу правду."),
            ("pt", "葡萄牙语", "Digite texto em português. Exemplo: Esta noite vou dizer a verdade."),
            ("it", "意大利语", "Inserisci testo in italiano. Esempio: Stasera dirò la verità."),
        ],
    ),
    (
        "中文方言",
        [
            ("sc", "四川话", "请输入四川话口语文本，例如：你啷个现在才回来嘛？"),
            ("yue", "粤语", "请输入粤语口语文本，例如：你而家喺边度？"),
            ("wu", "吴语", "请输入吴语口语文本，例如：侬今朝到啥地方去啦？"),
            ("dongbei", "东北话", "请输入东北话口语文本，例如：你咋现在才回来呢？"),
            ("henan", "河南话", "请输入河南话口语文本，例如：恁咋到现在才回来嘞？"),
            ("shaanxi", "陕西话", "请输入陕西话口语文本，例如：你咋这会儿才回来咧？"),
            ("shandong", "山东话", "请输入山东话口语文本，例如：你咋现在才回来啊？"),
            ("tianjin", "天津话", "请输入天津话口语文本，例如：嘛呢，你介才回来？"),
            ("minnan", "闽南话", "请输入闽南语口语文本，例如：你这马伫佗位？"),
        ],
    ),
]

LANGUAGE_LABELS = {
    value: label
    for _, options in LANGUAGE_GROUPS
    for value, label, _ in options
}


def _float_form(name: str, default: float) -> float:
    raw = request.form.get(name, "").strip()
    if not raw:
        return default
    return float(raw)


def _int_form(name: str, default: int) -> int:
    raw = request.form.get(name, "").strip()
    if not raw:
        return default
    return int(raw)


def _bool_form(name: str, default: bool = False) -> bool:
    if name not in request.form:
        return default
    return request.form.get(name) == "on"


def _page_context(**extra):
    context = {
        "references": list_references(),
        "tone_presets": list_tone_presets(),
        "language_groups": LANGUAGE_GROUPS,
        "language_labels": LANGUAGE_LABELS,
        "history": list_outputs(),
        "form": request.form,
    }
    context.update(extra)
    return context


def _generation_params(reference_key: str | None) -> dict:
    return {
        "text": request.form.get("text", ""),
        "mode": request.form.get("mode", "clone"),
        "language_key": request.form.get("language_key", "auto"),
        "reference_key": reference_key,
        "prompt_text": request.form.get("prompt_text", ""),
        "tone_key": request.form.get("tone_key", "natural"),
        "cfg_value": _float_form("cfg_value", 2.0),
        "inference_timesteps": _int_form("inference_timesteps", 10),
        "speed": _float_form("speed", 1.0),
        "normalize": _bool_form("normalize", False),
        "denoise": _bool_form("denoise", False),
        "retry_badcase": _bool_form("retry_badcase", False),
        "auto_split": _bool_form("auto_split", False),
        "max_segment_chars": _int_form("max_segment_chars", 180),
        "enable_control": _bool_form("enable_control", False),
        "control_instruction": request.form.get("control_instruction", ""),
        "output_name": request.form.get("output_name", ""),
    }


def _prepare_reference_key() -> str | None:
    reference_key = request.form.get("reference_key") or None
    upload = request.files.get("reference_upload")
    if upload and upload.filename:
        uploaded = save_uploaded_reference(upload.filename, upload)
        reference_key = uploaded.key
    return reference_key


def _set_job(job_id: str, **values) -> None:
    with _jobs_lock:
        _jobs.setdefault(job_id, {}).update(values)


def _get_job(job_id: str) -> dict | None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def _result_payload(result) -> dict:
    key = f"outputs:{result.filename}"
    return {
        "audio_key": key,
        "filename": result.filename,
        "duration": result.duration,
        "sample_rate": result.sample_rate,
        "speed": result.speed,
        "tone_label": result.tone_label,
        "language_label": LANGUAGE_LABELS.get(result.language_key, "自动识别"),
        "segment_count": result.segment_count,
    }


def _run_job(job_id: str, params: dict) -> None:
    def progress(percent: int, message: str) -> None:
        _set_job(job_id, percent=percent, message=message)

    with app.app_context():
        try:
            result = synthesize(**params, progress_callback=progress)
            _set_job(
                job_id,
                status="done",
                percent=100,
                message="生成完成",
                result=_result_payload(result),
            )
        except Exception as exc:
            _set_job(job_id, status="error", message=f"生成失败：{exc}")


@app.get("/")
def index():
    return render_template("index.html", **_page_context())


@app.post("/generate")
def generate():
    error = None
    result = None

    try:
        result = synthesize(**_generation_params(_prepare_reference_key()))
    except BusyError as exc:
        error = str(exc)
    except Exception as exc:
        error = f"生成失败：{exc}"

    return render_template("index.html", **_page_context(error=error, result=result))


@app.post("/generate_async")
def generate_async():
    try:
        params = _generation_params(_prepare_reference_key())
    except Exception as exc:
        return jsonify({"error": f"提交失败：{exc}"}), 400

    job_id = uuid.uuid4().hex
    _set_job(job_id, status="running", percent=1, message="任务已提交")
    thread = threading.Thread(target=_run_job, args=(job_id, params), daemon=True)
    thread.start()
    return jsonify({"job_id": job_id})


@app.post("/upload_reference")
def upload_reference():
    upload = request.files.get("reference_upload")
    if not upload or not upload.filename:
        return jsonify({"error": "请先选择参考音频文件"}), 400

    try:
        reference = save_uploaded_reference(upload.filename, upload)
    except Exception as exc:
        return jsonify({"error": f"上传失败：{exc}"}), 400

    return jsonify(
        {
            "key": reference.key,
            "name": reference.name,
            "duration": reference.duration,
            "sample_rate": reference.sample_rate,
        }
    )


@app.get("/progress/<job_id>")
def progress(job_id: str):
    job = _get_job(job_id)
    if not job:
        abort(404)
    result = job.get("result")
    if result and "audio_key" in result:
        result = dict(result)
        result["audio_url"] = url_for("audio", key=result["audio_key"])
        result["download_url"] = url_for("download", key=result["audio_key"])
        job["result"] = result
    return jsonify(job)


@app.get("/audio/<path:key>")
def audio(key: str):
    try:
        path = resolve_audio(key)
    except FileNotFoundError:
        abort(404)
    return send_file(path, mimetype="audio/wav", conditional=True)


@app.get("/download/<path:key>")
def download(key: str):
    try:
        path = resolve_audio(key)
    except FileNotFoundError:
        abort(404)
    return send_file(path, as_attachment=True, download_name=Path(path).name)


@app.post("/refresh")
def refresh():
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
