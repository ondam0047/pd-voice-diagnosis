"""인사이트 레이어 (M5) — 조음/언어 분석 → LLM 임상 코멘트.

APAC 음운변동 분류를 prompt로 제공해 패턴 해석을 요청한다.
룰 엔진으로 강제 분류하지 않으며, 사실 요약은 분류 없이 수치만 제시한다.
"""

from __future__ import annotations

import os

from modules.transcription import TranscriptionError, _get_client

APAC_TAXONOMY = (
    "발달적 음운변동: 전방화, 연구개음의 전설음화, 이완음화, 기식음화, 구개음화, 비음화, 유음화\n"
    "비전형적 음운변동: 어두초성 생략, 종성 생략, 비전형적 어중 단순화"
)


def summarize_articulation(articulation: dict) -> str:
    """분류 없는 사실 요약 텍스트 (프롬프트/오프라인 표시용)."""
    s = articulation["summary"]
    lines = [
        f"PCC(자음정확도): {articulation['pcc']}% "
        f"(정확 {s['correct_consonants']}/{s['total_consonants']}, 오류 {s['error_count']}, 첨가 {s['additions']})",
        f"위치별 오류: {articulation['position_errors']} / 전체 {articulation['position_total']}",
    ]
    pairs = []
    for tgt, row in articulation["confusion_matrix"].items():
        for prod, cnt in row.items():
            pairs.append((cnt, f"{tgt}→{prod}"))
    pairs.sort(reverse=True)
    if pairs:
        lines.append("주요 오류 패턴(목표→산출, 빈도순): "
                     + ", ".join(f"{p} ({c})" for c, p in pairs[:12]))
    low = [(acc, ph) for ph, acc in articulation["phoneme_accuracy"].items() if acc < 100]
    low.sort()
    if low:
        lines.append("정확도 낮은 음소: " + ", ".join(f"{ph} {acc}%" for acc, ph in low[:10]))
    return "\n".join(lines)


def summarize_language(language: dict) -> str:
    st = language["stats"]
    return (
        f"발화 수 {st['utterance_count']}, MLU-w {st['mlu_w']}, MLU-m {st['mlu_m']}, "
        f"TTR {st['ttr']}, NDW {st['ndw']}, TNW {st['tnw']}\n"
        f"품사: {st['semantic_counts']}\n"
        f"문장유형(추정): {st['sentence_types']}"
    )


def _build_prompt(articulation: dict | None, language: dict | None) -> tuple[str, str]:
    system = (
        "당신은 아동 말·언어 평가를 돕는 언어재활(언어치료) 전문가입니다. "
        "주어진 자동 분석 수치를 임상적으로 해석하되, 단정하지 말고 경향으로 기술하고, "
        "자동 분석의 한계(전사 오류·정렬 잡음 가능)를 전제로 임상가 검수를 권고하세요. "
        "아래 APAC 음운변동 분류를 참고해 관찰된 패턴이 어떤 변동에 해당할 수 있는지 제시하세요.\n\n"
        f"[APAC 음운변동 분류]\n{APAC_TAXONOMY}"
    )
    parts = []
    if articulation is not None:
        parts.append("[조음 분석 요약]\n" + summarize_articulation(articulation))
    if language is not None:
        parts.append("[언어 분석 요약]\n" + summarize_language(language))
    user = (
        "다음 자동 분석 결과를 바탕으로, ① 두드러진 조음·음운 오류 패턴과 가능한 음운변동 해석, "
        "② 어휘·구문(언어) 특징, ③ 임상적 제언을 간결한 한국어 불릿으로 정리하세요. "
        "확실하지 않은 부분은 검수 필요로 표시하세요.\n\n"
        + "\n\n".join(parts)
    )
    return system, user


def generate_insight(
    articulation: dict | None = None,
    language: dict | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> str:
    """LLM 임상 코멘트 생성. API 키 미설정 시 TranscriptionError."""
    if articulation is None and language is None:
        raise TranscriptionError("분석 결과가 없습니다.")
    client = _get_client(api_key)
    model = model or os.getenv("INSIGHT_MODEL", "gpt-4o-mini")
    system, user = _build_prompt(articulation, language)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.3,
        )
    except Exception as e:
        raise TranscriptionError(f"인사이트 생성 실패: {e}") from e
    return (resp.choices[0].message.content or "").strip()
