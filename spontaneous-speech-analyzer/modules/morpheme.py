"""형태소 분석 래퍼 (kiwipiepy).

자발화(발화 리스트)를 입력받아 언어 분석 지표를 산출한다.

지표 정의 (M1, 투명성 우선):
- MLU-w (평균 어절 길이): 총 어절 수 / 발화 수
- MLU-m (평균 형태소 길이): 총 형태소 수 / 발화 수
- TNW (총 어절 수): 모든 발화의 어절 합
- NDW (서로 다른 어절 수): 어절 type 수
- TTR (어휘 다양도): NDW / TNW (어절 기준)
- 문법형태소 분포: 조사/어미/접사 범주별 빈도

* '낱말' 단위는 공백 기준 어절(eojeol)을 사용한다. 임상 현장의 낱말 정의가
  기관마다 다르므로, 본 도구는 재현 가능한 어절 기준을 명시적으로 채택한다.
"""

from __future__ import annotations

from collections import Counter

from kiwipiepy import Kiwi

# --- 품사 태그 그룹 ---
JOSA_TAGS = {"JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC"}
EOMI_TAGS = {"EP", "EF", "EC", "ETN", "ETM"}
AFFIX_TAGS = {"XPN", "XSN", "XSV", "XSA", "XR"}
NOUN_TAGS = {"NNG", "NNP", "NNB", "NR", "NP"}
VERB_TAGS = {"VV", "VA", "VX", "VCP", "VCN"}
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


def _major_class(tag: str) -> str:
    """품사 대분류(한국어 라벨) 반환."""
    if tag in NOUN_TAGS:
        return "체언(명사류)"
    if tag in VERB_TAGS:
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
    """kiwipiepy 기반 자발화 형태소 분석기."""

    def __init__(self) -> None:
        self.kiwi = Kiwi()

    def analyze(self, utterances: list[str]) -> dict:
        """발화 리스트를 분석해 발화별 상세 + 전체 통계를 반환한다."""
        clean = [u.strip() for u in utterances if u and u.strip()]

        per_utterance: list[dict] = []
        all_eojeols: list[str] = []
        word_counter: Counter[str] = Counter()
        gram_counter: Counter[str] = Counter()
        pos_counter: Counter[str] = Counter()
        total_morphemes = 0
        total_words = 0

        for utt in clean:
            tokens = self.kiwi.tokenize(utt)
            morphs = [t for t in tokens if t.tag not in PUNCT_TAGS]
            eojeols = utt.split()

            total_words += len(eojeols)
            total_morphemes += len(morphs)
            all_eojeols.extend(eojeols)
            word_counter.update(eojeols)

            for t in morphs:
                pos_counter[_major_class(t.tag)] += 1
                if t.tag in GRAMMATICAL_TAGS:
                    gram_counter[TAG_LABELS.get(t.tag, t.tag)] += 1

            per_utterance.append({
                "text": utt,
                "words": len(eojeols),
                "morphemes": len(morphs),
                "tokens": [
                    {"form": t.form, "tag": t.tag,
                     "label": TAG_LABELS.get(t.tag, _major_class(t.tag))}
                    for t in morphs
                ],
            })

        n = len(clean)
        tnw = total_words
        ndw = len(set(all_eojeols))

        stats = {
            "utterance_count": n,
            "total_morphemes": total_morphemes,
            "total_words": total_words,
            "mlu_w": round(total_words / n, 2) if n else 0.0,
            "mlu_m": round(total_morphemes / n, 2) if n else 0.0,
            "tnw": tnw,
            "ndw": ndw,
            "ttr": round(ndw / tnw, 3) if tnw else 0.0,
            "grammatical_morphemes": dict(gram_counter.most_common()),
            "pos_distribution": dict(pos_counter.most_common()),
            "word_freq": word_counter.most_common(20),
        }

        return {"stats": stats, "utterances": per_utterance}
