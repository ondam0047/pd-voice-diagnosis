"""음성 전사 (OpenAI Whisper API) — 표준어(목표어) 전사.

M2: transcribe_target — Whisper로 표준어 초안 전사, Whisper 세그먼트 타임스탬프로
발화 단위 자동 분할.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

WHISPER_SIZE_LIMIT = 25 * 1024 * 1024  # Whisper API 25MB 제한


class TranscriptionError(Exception):
    """전사 과정에서 발생하는 사용자 대응 가능한 오류."""


def _get_client():
    try:
        from openai import OpenAI
    except ImportError as e:  # pragma: no cover
        raise TranscriptionError(
            "openai 패키지가 설치되어 있지 않습니다. `pip install openai`"
        ) from e
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not key or key.startswith("sk-...") or key.lower() == "your_key_here":
        raise TranscriptionError(
            "OPENAI_API_KEY가 설정되지 않았습니다. .env 파일에 키를 입력하세요."
        )
    return OpenAI(api_key=key)


def _seg_attr(seg, name):
    if isinstance(seg, dict):
        return seg.get(name)
    return getattr(seg, name, None)


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
    file_name: str, audio_bytes: bytes, language: str = "ko"
) -> list[dict]:
    """음성 → 표준어 전사(발화 리스트). 각 항목: index, start, end, text."""
    if not audio_bytes:
        raise TranscriptionError("빈 오디오 파일입니다.")
    if len(audio_bytes) > WHISPER_SIZE_LIMIT:
        raise TranscriptionError(
            "파일이 25MB를 초과합니다 (Whisper API 제한). 파일을 분할해 주세요."
        )
    client = _get_client()
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
    return _parse_segments(resp)


def format_ts(seconds: float) -> str:
    """초 → mm:ss."""
    m, s = divmod(int(round(seconds)), 60)
    return f"{m:02d}:{s:02d}"
