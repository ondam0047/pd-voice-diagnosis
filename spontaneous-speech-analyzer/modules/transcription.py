"""음성 전사 (OpenAI Whisper API) — 표준어(목표어) 전사.

M2: transcribe_target — Whisper로 표준어 초안 전사, Whisper 세그먼트 타임스탬프로
발화 단위 자동 분할.
"""

from __future__ import annotations

import base64
import io
import os
import re
import shutil
import subprocess
import tempfile
import wave

from dotenv import load_dotenv

load_dotenv()

WHISPER_SIZE_LIMIT = 25 * 1024 * 1024  # Whisper API 25MB 제한


class TranscriptionError(Exception):
    """전사 과정에서 발생하는 사용자 대응 가능한 오류."""


def _get_client(api_key: str | None = None):
    try:
        from openai import OpenAI
    except ImportError as e:  # pragma: no cover
        raise TranscriptionError(
            "openai 패키지가 설치되어 있지 않습니다. `pip install openai`"
        ) from e
    key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
    if not key or key.startswith("sk-...") or key.lower() == "your_key_here":
        raise TranscriptionError(
            "OpenAI API 키가 없습니다. 사이드바에 키를 입력하거나 .env에 설정하세요."
        )
    return OpenAI(api_key=key)


def _seg_attr(seg, name):
    if isinstance(seg, dict):
        return seg.get(name)
    return getattr(seg, name, None)


_SENT_SPLIT = re.compile(r"(?<=[.!?。])\s+")


def _split_by_sentence(segments: list[dict]) -> list[dict]:
    """한 세그먼트에 여러 문장이 묶인 경우 종결부호(. ! ?) 기준으로 발화 분리.

    문장별 정확한 타임스탬프가 없으므로 글자 수 비례로 시간을 분배한다.
    """
    out: list[dict] = []
    for s in segments:
        text = (s.get("text") or "").strip()
        parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
        if len(parts) <= 1:
            out.append({"start": float(s.get("start", 0.0)),
                        "end": float(s.get("end", 0.0)), "text": text})
            continue
        total = sum(len(p) for p in parts) or 1
        start = float(s.get("start", 0.0))
        dur = max(0.0, float(s.get("end", start)) - start)
        acc = start
        for p in parts:
            seg_end = acc + dur * (len(p) / total)
            out.append({"start": acc, "end": seg_end, "text": p})
            acc = seg_end
    for i, s in enumerate(out, 1):
        s["index"] = i
    return out


def _parse_segments(resp) -> list[dict]:
    """Whisper verbose_json 응답 → 발화(세그먼트) 리스트."""
    segments = _seg_attr(resp, "segments") or []
    out: list[dict] = []
    for seg in segments:
        text = (_seg_attr(seg, "text") or "").strip()
        if not text:
            continue
        out.append({
            "index": len(out) + 1,
            "start": float(_seg_attr(seg, "start") or 0.0),
            "end": float(_seg_attr(seg, "end") or 0.0),
            "text": text,
        })
    if not out:  # 세그먼트가 없으면 전체 텍스트를 단일 발화로
        full = (_seg_attr(resp, "text") or "").strip()
        if full:
            out.append({"index": 1, "start": 0.0, "end": 0.0, "text": full})
    return out


def transcribe_target(
    file_name: str, audio_bytes: bytes, language: str = "ko", api_key: str | None = None
) -> list[dict]:
    """음성 → 표준어 전사(발화 리스트). 각 항목: index, start, end, text."""
    if not audio_bytes:
        raise TranscriptionError("빈 오디오 파일입니다.")
    if len(audio_bytes) > WHISPER_SIZE_LIMIT:
        raise TranscriptionError(
            "파일이 25MB를 초과합니다 (Whisper API 제한). 파일을 분할해 주세요."
        )
    client = _get_client(api_key)
    model = os.getenv("WHISPER_MODEL", "whisper-1")
    try:
        resp = client.audio.transcriptions.create(
            model=model,
            file=(file_name, audio_bytes),
            response_format="verbose_json",
            language=language,
        )
    except Exception as e:  # API/네트워크 오류
        raise TranscriptionError(f"전사 실패: {e}") from e
    return _split_by_sentence(_parse_segments(resp))


def format_ts(seconds: float) -> str:
    """초 → mm:ss."""
    m, s = divmod(int(round(seconds)), 60)
    return f"{m:02d}:{s:02d}"


# ===================== 산출형 전사 (GPT-4o audio) =====================

def _slice_wav(audio_bytes: bytes, start: float, end: float) -> bytes:
    """wav 바이트에서 [start, end] 구간을 잘라 wav 바이트로 반환 (ffmpeg 불필요)."""
    with wave.open(io.BytesIO(audio_bytes), "rb") as w:
        fr, nch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        w.setpos(min(int(start * fr), w.getnframes()))
        frames = w.readframes(max(0, int((end - start) * fr)))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as o:
        o.setnchannels(nch)
        o.setsampwidth(sw)
        o.setframerate(fr)
        o.writeframes(frames)
    return buf.getvalue()


def _ffmpeg_exe() -> str:
    """시스템 ffmpeg 우선, 없으면 pip 설치 정적 ffmpeg(imageio-ffmpeg)."""
    sys_ff = shutil.which("ffmpeg")
    if sys_ff:
        return sys_ff
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # pragma: no cover
        raise TranscriptionError(
            "ffmpeg를 찾을 수 없습니다. `pip install imageio-ffmpeg` 또는 ffmpeg 설치."
        ) from e


def slice_audio(audio_bytes: bytes, file_name: str, start: float, end: float):
    """오디오 구간 분할 → (wav_bytes, 'wav').

    wav는 표준 라이브러리로 무손실 분할(ffmpeg 불필요). mp3/m4a는 ffmpeg 바이너리를
    직접 호출(ffprobe 불필요). imageio-ffmpeg가 설치돼 있으면 별도 설치 없이 동작.
    """
    ext = file_name.rsplit(".", 1)[-1].lower()
    if ext == "wav":
        return _slice_wav(audio_bytes, start, end), "wav"
    ff = _ffmpeg_exe()
    in_path = out_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tf:
            tf.write(audio_bytes)
            in_path = tf.name
        out_path = in_path + ".out.wav"
        # WAV는 시킹이 필요해 pipe 대신 파일로 출력(헤더 크기 정상 기록)
        cmd = [
            ff, "-y", "-hide_banner", "-loglevel", "error",
            "-i", in_path, "-ss", str(start), "-t", str(max(0.0, end - start)),
            "-ac", "1", "-ar", "16000", out_path,
        ]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0 or not os.path.exists(out_path):
            raise TranscriptionError(
                f"{ext} 분할 실패: {proc.stderr.decode(errors='ignore')[:200]}")
        with open(out_path, "rb") as f:
            return f.read(), "wav"
    finally:
        for p in (in_path, out_path):
            if p and os.path.exists(p):
                os.remove(p)


def _build_produced_prompt(few_shot: dict) -> tuple[str, str]:
    examples = few_shot.get("phonetic_transcription_examples", [])
    instruction = few_shot.get("instruction", "")
    ex_text = "\n".join(
        f'- "{e["standard"]}" → "{e["produced"]}"' for e in examples
    )
    system = (
        "당신은 아동 말소리를 들리는 실제 발음 그대로 한글로 전사하는 전문가입니다. "
        + instruction
    )
    user = (
        "다음 예시처럼 표준어로 보정하지 말고, 오디오에서 들리는 실제 발음대로 "
        "한글로만 한 줄 전사하세요. 설명 없이 전사 결과만 출력합니다.\n\n"
        f"[예시]\n{ex_text}\n\n[전사할 발화 → 산출형]:"
    )
    return system, user


def transcribe_produced(
    file_name: str, audio_bytes: bytes, few_shot: dict,
    language: str = "ko", api_key: str | None = None,
) -> str:
    """음성 구간 → 산출형(실제 발음) 한글 전사. GPT-4o audio + few-shot."""
    if not audio_bytes:
        raise TranscriptionError("빈 오디오입니다.")
    audio_format = file_name.rsplit(".", 1)[-1].lower()
    if audio_format not in ("wav", "mp3"):
        raise TranscriptionError(
            f"GPT-4o audio는 wav/mp3만 지원합니다 (현재: {audio_format})."
        )
    client = _get_client(api_key)
    model = os.getenv("GPT_AUDIO_MODEL", "gpt-4o-audio-preview")
    system, user = _build_produced_prompt(few_shot)
    audio_b64 = base64.b64encode(audio_bytes).decode()
    try:
        resp = client.chat.completions.create(
            model=model,
            modalities=["text"],
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "text", "text": user},
                    {"type": "input_audio",
                     "input_audio": {"data": audio_b64, "format": audio_format}},
                ]},
            ],
        )
    except Exception as e:
        raise TranscriptionError(f"산출형 전사 실패: {e}") from e
    return (resp.choices[0].message.content or "").strip()
