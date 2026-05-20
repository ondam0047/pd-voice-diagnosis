"""한글 음절 → 자모(초성/중성/종성) 분리.

위치 정보(초성/중성/종성, 음절 인덱스)를 보존하기 위해 자모 라이브러리의 모호함 대신
유니코드 산술 분해를 사용한다.
"""

from __future__ import annotations

CHO = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
JUNG = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
JONG = ["", *list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")]

_BASE = 0xAC00
_N = 11172


def decompose_char(ch: str) -> tuple[str, str, str] | None:
    """한글 음절 → (초성, 중성, 종성). 종성 없으면 ''. 한글 아니면 None."""
    code = ord(ch) - _BASE
    if 0 <= code < _N:
        cho = code // 588
        jung = (code % 588) // 28
        jong = code % 28
        return CHO[cho], JUNG[jung], JONG[jong]
    return None


def word_phonemes(word: str) -> list[dict]:
    """어절 → 음소 리스트(순서 보존). 각 항목: jamo, kind(초성/중성/종성), syllable(음절 인덱스)."""
    out: list[dict] = []
    syl = 0
    for ch in word:
        dec = decompose_char(ch)
        if dec is None:
            continue
        cho, jung, jong = dec
        out.append({"jamo": cho, "kind": "초성", "syllable": syl})
        out.append({"jamo": jung, "kind": "중성", "syllable": syl})
        if jong:
            out.append({"jamo": jong, "kind": "종성", "syllable": syl})
        syl += 1
    return out
