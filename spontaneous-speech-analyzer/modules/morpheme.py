"""형태소 분석 래퍼 (kiwipiepy).

자발화(발화 리스트)를 입력받아 언어 분석 지표를 산출한다.

낱말(word) 단위 정의 — 품사 기준 (세종 품사 체계):
- 낱말 = 체언 + 용언 (내용어)
  - 체언: NNG(일반명사) NNP(고유명사) NNB(의존명사) NR(수사) NP(대명사)
  - 용언: VV(동사) VA(형용사) VX(보조용언) VCP(긍정지정사) VCN(부정지정사)

지표 정의:
- MLU-w (평균 낱말 길이): 총 낱말 수 / 발화 수  (낱말 = 체언+용언)
- MLU-m (평균 형태소 길이): 총 형태소 수 / 발화 수
- TNW (총 낱말 수): 모든 발화의 체언+용언 합
- NDW (서로 다른 낱말 수): 내용어 type 수 (어형 기준)
- TTR (어휘 다양도): NDW / TNW
- 문법형태소 분포: 조사/어미/접사 범주별 빈도
"""

from __future__ import annotations

from collections import Counter

from kiwipiepy import Kiwi

# --- 품사 태그 그룹 ---
# 체언 / 용언 (낱말 단위)
CHEEON_TAGS = {"NNG", "NNP", "NNB", "NR", "NP"}
YONGEON_TAGS = {"VV", "VA", "VX", "VCP", "VCN"}
CONTENT_TAGS = CHEEON_TAGS | YONGEON_TAGS  # 낱말 = 체언 + 용언

JOSA_TAGS = {"JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC"}
EOMI_TAGS = {"EP", "EF", "EC", "ETN", "ETM"}
AFFIX_TAGS = {"XPN", "XSN", "XSV", "XSA", "XR"}
MODIFIER_TAGS = {"MM", "MAG", "MAJ"}
# 형태소 수 계산에서 제외할 문장부호/기호 태그
PUNCT_TAGS = {"SF", "SP", "SS", "SE", "SO", "SW", "SB"}

# 문법형태소 = 조사 + 어미 + 접사
GRAMMATICAL_TAGS = JOSA_TAGS | EOMI_TAGS | AFFIX_TAGS

# 태그 → 한국어 라벨
TAG_LABELS = {
    # 조사
    "JKS": "주격조사", "JKC": "보격조사", "JKG": "관형격조사", "JKO": "목적격조사",
    "JKB": "부사격조사", "JKV": "호격조사", "JKQ": "인용격조사",
    "JX": "보조사", "JC": "접속조사",
    # 어미
    "EP": "선어말어미", "EF": "종결어미", "EC": "연결어미",
    "ETN": "명사형전성어미", "ETM": "관형형전성어미",
    # 접사
    "XPN": "체언접두사", "XSN": "명사파생접미사",
    "XSV": "동사파생접미사", "XSA": "형용사파생접미사", "XR": "어근",
}


def _word_class(tag: str) -> str | None:
    """낱말 범주(체언/용언) 반환. 낱말이 아니면 None."""
    if tag in CHEEON_TAGS:
        return "체언"
    if tag in YONGEON_TAGS:
        return "용언"
    return None


def _major_class(tag: str) -> str:
    """품사 대분류(한국어 라벨) 반환."""
    if tag in CHEEON_TAGS:
        return "체언(명사류)"
    if tag in YONGEON_TAGS:
        return "용언(동사/형용사)"
    if tag in MODIFIER_TAGS:
        return "수식언(관형사/부사)"
    if tag == "IC":
        return "독립언(감탄사)"
    if tag in JOSA_TAGS:
        return "관계언(조사)"
    if tag in EOMI_TAGS:
        return "어미"
    if tag in AFFIX_TAGS:
        return "접사"
    return "기타"


class MorphemeAnalyzer:
    """kiwipiepy 기반 자발화 형태소 분석기. 낱말 단위는 품사(체언/용언) 기준."""

    def __init__(self) -> None:
        self.kiwi = Kiwi()

    def analyze(self, utterances: list[str]) -> dict:
        """발화 리스트를 분석해 발화별 상세 + 전체 통계를 반환한다."""
        clean = [u.strip() for u in utterances if u and u.strip()]

        per_utterance: list[dict] = []
        word_types: set[tuple[str, str]] = set()          # (어형, 범주)
        cheeon_types: set[str] = set()
        yongeon_types: set[str] = set()
        word_counter: Counter[tuple[str, str]] = Counter()
        gram_counter: Counter[str] = Counter()
        pos_counter: Counter[str] = Counter()
        total_morphemes = 0
        total_words = 0
        cheeon_count = 0
        yongeon_count = 0

        for utt in clean:
            tokens = self.kiwi.tokenize(utt)
            morphs = [t for t in tokens if t.tag not in PUNCT_TAGS]

            u_cheeon = 0
            u_yongeon = 0
            for t in morphs:
                pos_counter[_major_class(t.tag)] += 1
                if t.tag in GRAMMATICAL_TAGS:
                    gram_counter[TAG_LABELS.get(t.tag, t.tag)] += 1
                wc = _word_class(t.tag)
                if wc == "체언":
                    u_cheeon += 1
                    cheeon_types.add(t.form)
                    word_types.add((t.form, "체언"))
                    word_counter[(t.form, "체언")] += 1
                elif wc == "용언":
                    u_yongeon += 1
                    yongeon_types.add(t.form)
                    word_types.add((t.form, "용언"))
                    word_counter[(t.form, "용언")] += 1

            u_words = u_cheeon + u_yongeon
            total_morphemes += len(morphs)
            total_words += u_words
            cheeon_count += u_cheeon
            yongeon_count += u_yongeon

            per_utterance.append({
                "text": utt,
                "words": u_words,
                "cheeon": u_cheeon,
                "yongeon": u_yongeon,
                "morphemes": len(morphs),
                "tokens": [
                    {"form": t.form, "tag": t.tag,
                     "label": TAG_LABELS.get(t.tag, _major_class(t.tag)),
                     "word_class": _word_class(t.tag)}
                    for t in morphs
                ],
            })

        n = len(clean)
        tnw = total_words
        ndw = len(word_types)

        stats = {
            "utterance_count": n,
            "total_morphemes": total_morphemes,
            "total_words": total_words,            # TNW (체언+용언)
            "cheeon_count": cheeon_count,
            "yongeon_count": yongeon_count,
            "mlu_w": round(total_words / n, 2) if n else 0.0,
            "mlu_m": round(total_morphemes / n, 2) if n else 0.0,
            "tnw": tnw,
            "ndw": ndw,
            "ttr": round(ndw / tnw, 3) if tnw else 0.0,
            "cheeon_ndw": len(cheeon_types),
            "yongeon_ndw": len(yongeon_types),
            "grammatical_morphemes": dict(gram_counter.most_common()),
            "pos_distribution": dict(pos_counter.most_common()),
            "word_freq": [
                {"word": form, "word_class": wc, "count": cnt}
                for (form, wc), cnt in word_counter.most_common(20)
            ],
        }

        return {"stats": stats, "utterances": per_utterance}
