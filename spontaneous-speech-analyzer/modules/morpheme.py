"""형태소 분석 래퍼 (kiwipiepy).

자발화(발화 리스트)를 입력받아 언어 분석 지표를 산출한다.
낱말/품사 분류는 임상 자발화 분석 정답지 양식에 맞춘다.

낱말(word) 정의 — 내용어 = 체언 + 용언 + 수식언 + 독립언
  - 체언: 명사(NNG/NNP/NNB/NR), 대명사(NP)
  - 용언: 동사(VV/VX), 형용사(VA/VCN)        ※ 용언은 기본형으로 표기
  - 수식언: 부사(MAG/MAJ), 관형사(MM)
  - 독립언: 감탄사(IC)
  * 지정사 '이다'(VCP)는 서술격조사로 보아 낱말에서 제외(정답지 방식).

지표 정의:
- MLU-w (평균 낱말 길이): 총 낱말 수 / 발화 수
- MLU-m (평균 형태소 길이): 총 형태소 수 / 발화 수
- TNW(총 낱말) / NDW(서로 다른 낱말) / TTR(=NDW/TNW)
- 의미 영역 세분화: 명사·대명사·동사·형용사·부사·관형사·독립언
"""

from __future__ import annotations

from collections import Counter

from kiwipiepy import Kiwi

# --- 의미 영역(낱말) 품사 매핑 ---
SEMANTIC_MAP = {
    "NNG": "명사", "NNP": "명사", "NNB": "명사", "NR": "명사",
    "NP": "대명사",
    "VV": "동사", "VX": "동사",
    "VA": "형용사", "VCN": "형용사",
    "MAG": "부사", "MAJ": "부사",
    "MM": "관형사",
    "IC": "독립언",
}
SEMANTIC_ORDER = ["명사", "대명사", "동사", "형용사", "부사", "관형사", "독립언"]
BROAD_OF = {
    "명사": "체언", "대명사": "체언",
    "동사": "용언", "형용사": "용언",
    "부사": "수식언", "관형사": "수식언",
    "독립언": "독립언",
}
PREDICATE_CATS = {"동사", "형용사"}  # 기본형(+다) 표기 대상

# --- 문법형태소 태그 ---
JOSA_TAGS = {"JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC"}
EOMI_TAGS = {"EP", "EF", "EC", "ETN", "ETM"}
AFFIX_TAGS = {"XPN", "XSN", "XSV", "XSA", "XR"}
GRAMMATICAL_TAGS = JOSA_TAGS | EOMI_TAGS | AFFIX_TAGS
# 형태소 수 계산에서 제외할 문장부호/기호 태그
PUNCT_TAGS = {"SF", "SP", "SS", "SE", "SO", "SW", "SB"}

TAG_LABELS = {
    "JKS": "주격조사", "JKC": "보격조사", "JKG": "관형격조사", "JKO": "목적격조사",
    "JKB": "부사격조사", "JKV": "호격조사", "JKQ": "인용격조사",
    "JX": "보조사", "JC": "접속조사",
    "EP": "선어말어미", "EF": "어말어미(종결)", "EC": "연결어미",
    "ETN": "전성어미(명사형)", "ETM": "전성어미(관형형)",
    "XPN": "체언접두사", "XSN": "명사파생접미사",
    "XSV": "동사파생접미사", "XSA": "형용사파생접미사", "XR": "어근",
}


def _headword(form: str, cat: str | None) -> str:
    """낱말 표제어. 용언(동사/형용사)은 기본형(어간+다)으로."""
    if cat in PREDICATE_CATS:
        return form + "다"
    return form


class MorphemeAnalyzer:
    """kiwipiepy 기반 자발화 형태소 분석기 (정답지 양식 의미 영역 세분화)."""

    def __init__(self) -> None:
        self.kiwi = Kiwi()

    def analyze(self, utterances: list[str]) -> dict:
        clean = [u.strip() for u in utterances if u and u.strip()]

        per_utterance: list[dict] = []
        word_types: set[tuple[str, str]] = set()          # (표제어, 세부품사)
        type_by_cat: dict[str, set[str]] = {c: set() for c in SEMANTIC_ORDER}
        word_counter: Counter[tuple[str, str]] = Counter()
        sem_counter: Counter[str] = Counter()
        gram_counter: Counter[str] = Counter()
        total_morphemes = 0
        total_words = 0

        for utt in clean:
            tokens = self.kiwi.tokenize(utt)
            # kiwi는 동사/형용사 불규칙 활용에 'VA-I' 같은 접미를 붙이므로 기저 태그로 정규화
            morphs = [t for t in tokens if t.tag.split("-", 1)[0] not in PUNCT_TAGS]
            total_morphemes += len(morphs)

            u_sem: Counter[str] = Counter()
            u_tokens = []
            for t in morphs:
                base = t.tag.split("-", 1)[0]
                cat = SEMANTIC_MAP.get(base)
                if base in GRAMMATICAL_TAGS:
                    gram_counter[TAG_LABELS.get(base, base)] += 1
                if cat:  # 낱말(내용어)
                    head = _headword(t.form, cat)
                    u_sem[cat] += 1
                    sem_counter[cat] += 1
                    type_by_cat[cat].add(head)
                    word_types.add((head, cat))
                    word_counter[(head, cat)] += 1
                u_tokens.append({
                    "form": t.form, "tag": t.tag,
                    "headword": _headword(t.form, cat) if cat else t.form,
                    "category": cat, "broad": BROAD_OF.get(cat) if cat else None,
                })

            u_words = sum(u_sem.values())
            total_words += u_words
            per_utterance.append({
                "text": utt,
                "words": u_words,
                "morphemes": len(morphs),
                "semantic": {c: u_sem.get(c, 0) for c in SEMANTIC_ORDER},
                "tokens": u_tokens,
            })

        n = len(clean)
        tnw = total_words
        ndw = len(word_types)

        broad_counts: Counter[str] = Counter()
        for cat in SEMANTIC_ORDER:
            broad_counts[BROAD_OF[cat]] += sem_counter.get(cat, 0)

        stats = {
            "utterance_count": n,
            "total_morphemes": total_morphemes,
            "total_words": total_words,
            "mlu_w": round(total_words / n, 2) if n else 0.0,
            "mlu_m": round(total_morphemes / n, 2) if n else 0.0,
            "tnw": tnw,
            "ndw": ndw,
            "ttr": round(ndw / tnw, 3) if tnw else 0.0,
            "semantic_counts": {c: sem_counter.get(c, 0) for c in SEMANTIC_ORDER},
            "semantic_ndw": {c: len(type_by_cat[c]) for c in SEMANTIC_ORDER},
            "broad_counts": dict(broad_counts),
            "grammatical_morphemes": dict(gram_counter.most_common()),
            "word_freq": [
                {"word": head, "category": cat, "count": cnt}
                for (head, cat), cnt in word_counter.most_common(20)
            ],
        }

        return {"stats": stats, "utterances": per_utterance}
