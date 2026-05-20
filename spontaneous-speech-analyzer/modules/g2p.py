"""g2p 래퍼 — 목표어(표준 철자) → 발음형 변환.

g2pkk(메캡 불필요) 우선, 없으면 g2pk 사용.
"""

from __future__ import annotations


def _load_g2p():
    try:
        from g2pkk import G2p
        return G2p()
    except Exception:
        from g2pk import G2p
        return G2p()


class G2PConverter:
    """목표어 → 발음형 변환기."""

    def __init__(self) -> None:
        self._g2p = _load_g2p()

    def to_pronunciation(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        try:
            return self._g2p(text)
        except Exception:
            return text  # 변환 실패 시 원문 유지
