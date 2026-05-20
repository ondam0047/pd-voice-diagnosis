"""📝 언어 분석 — MLU/TTR/NDW (M1: 텍스트 입력 모드).

낱말 = 체언+용언+수식언+독립언. 의미 영역(명사/대명사·동사/형용사·부사/관형사·독립언) 세분화.
"""

import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

# 프로젝트 루트를 import 경로에 추가 (페이지 직접 실행 대비)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.morpheme import BROAD_OF, SEMANTIC_ORDER, MorphemeAnalyzer  # noqa: E402

st.set_page_config(page_title="언어 분석", page_icon="📝", layout="wide")

# 예시 발화 (정답지 표준어 형태 기반)
SAMPLE_UTTERANCES = """엄마랑 아빠랑 같이 큰집에 갔어요
토끼가 추운데 죽어서 너무 슬펐어요
시골이라서 연을 날릴 수 있어요
국이 너무 뜨거워서 못 먹었어요
그러면 우리 같이 영화 보러 가요"""


@st.cache_resource(show_spinner="형태소 분석기 로딩 중…")
def get_analyzer() -> MorphemeAnalyzer:
    return MorphemeAnalyzer()


st.title("📝 언어 분석")
st.caption("MLU-w · MLU-m · TTR · NDW · TNW  ·  낱말 = 체언 + 용언 + 수식언 + 독립언")

# --- 입력 방식 선택 ---
input_mode = st.radio(
    "입력 방식",
    ["텍스트 직접 입력", "음성 업로드 (M2 예정)"],
    horizontal=True,
)

if input_mode == "음성 업로드 (M2 예정)":
    st.info("음성 업로드 → Whisper 자동 전사 기능은 M2에서 제공됩니다. 지금은 텍스트 입력을 사용하세요.")
    st.file_uploader("음성 파일 (.mp3, .wav, .m4a)", type=["mp3", "wav", "m4a"], disabled=True)
    st.stop()

st.markdown("**발화를 한 줄에 하나씩 입력하세요.** (한 줄 = 한 발화)")
st.caption("정확한 지표를 위해 표준어로 정규화하고 반복·수정·간투사(마디)는 제외한 전사를 권장합니다.")

if st.button("예시 발화 불러오기"):
    st.session_state["lang_text"] = SAMPLE_UTTERANCES

text = st.text_area(
    "발화 입력",
    key="lang_text",
    height=220,
    placeholder="예)\n엄마랑 아빠랑 같이 큰집에 갔어요\n그러면 우리 같이 영화 보러 가요",
)

run = st.button("분석 실행", type="primary")

if run:
    utterances = [line for line in text.splitlines() if line.strip()]
    if not utterances:
        st.warning("발화를 한 줄 이상 입력하세요.")
        st.stop()

    analyzer = get_analyzer()
    result = analyzer.analyze(utterances)
    stats = result["stats"]

    # --- 핵심 지표 ---
    st.subheader("핵심 지표")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("발화 수", stats["utterance_count"])
    c2.metric("MLU-w (평균 낱말)", stats["mlu_w"])
    c3.metric("MLU-m (평균 형태소)", stats["mlu_m"])
    c4.metric("TTR (어휘 다양도)", stats["ttr"])

    c5, c6, c7 = st.columns(3)
    c5.metric("TNW (총 낱말 수)", stats["tnw"])
    c6.metric("NDW (서로 다른 낱말)", stats["ndw"])
    c7.metric("총 형태소 수", stats["total_morphemes"])

    st.caption(
        "낱말 = 체언+용언+수식언+독립언 (용언은 기본형). "
        "MLU-w = 총 낱말 / 발화 수 · MLU-m = 총 형태소 / 발화 수 · TTR = NDW / TNW."
    )

    st.divider()

    # --- 의미 영역 세분화 ---
    st.subheader("의미 영역 (품사 세분화)")
    b = stats["broad_counts"]
    bc1, bc2, bc3, bc4 = st.columns(4)
    bc1.metric("체언", b.get("체언", 0))
    bc2.metric("용언", b.get("용언", 0))
    bc3.metric("수식언", b.get("수식언", 0))
    bc4.metric("독립언", b.get("독립언", 0))

    sem_df = pd.DataFrame({
        "품사": SEMANTIC_ORDER,
        "대분류": [BROAD_OF[c] for c in SEMANTIC_ORDER],
        "총 낱말": [stats["semantic_counts"][c] for c in SEMANTIC_ORDER],
        "서로 다른 낱말": [stats["semantic_ndw"][c] for c in SEMANTIC_ORDER],
    })
    fig = px.bar(
        sem_df.melt(id_vars=["품사", "대분류"], var_name="구분", value_name="빈도"),
        x="품사", y="빈도", color="구분", barmode="group", text="빈도",
        category_orders={"품사": SEMANTIC_ORDER},
        color_discrete_map={"총 낱말": "#4C78A8", "서로 다른 낱말": "#9ECAE1"},
    )
    fig.update_layout(xaxis_title="", legend_title="", height=380)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(sem_df, use_container_width=True, hide_index=True)

    st.divider()

    # --- 문법형태소 분포 ---
    st.subheader("문법형태소 분포")
    gram = stats["grammatical_morphemes"]
    if gram:
        gram_df = pd.DataFrame(
            {"문법형태소": list(gram.keys()), "빈도": list(gram.values())}
        )
        gfig = px.bar(
            gram_df, x="문법형태소", y="빈도", text="빈도",
            color="빈도", color_continuous_scale="Blues",
        )
        gfig.update_layout(xaxis_title="", coloraxis_showscale=False, height=360)
        st.plotly_chart(gfig, use_container_width=True)
    else:
        st.info("문법형태소(조사·어미·접사)가 검출되지 않았습니다.")

    st.divider()

    # --- 발화별 상세 ---
    st.subheader("발화별 상세")
    detail_df = pd.DataFrame(
        [
            {"#": i + 1, "발화": u["text"], "낱말": u["words"], "형태소": u["morphemes"],
             **{c: u["semantic"][c] for c in SEMANTIC_ORDER}}
            for i, u in enumerate(result["utterances"])
        ]
    )
    st.dataframe(detail_df, use_container_width=True, hide_index=True)

    with st.expander("형태소 분해 보기 (낱말=기본형/품사 표시)"):
        for i, u in enumerate(result["utterances"]):
            st.markdown(f"**{i + 1}. {u['text']}**")
            parts = []
            for t in u["tokens"]:
                if t["category"]:
                    parts.append(f"`{t['headword']}`⟨{t['category']}⟩")
                else:
                    parts.append(f"`{t['form']}`/{t['tag']}")
            st.markdown("  ".join(parts))

    if stats["word_freq"]:
        with st.expander("낱말 빈도 (기본형, 상위 20)"):
            wf_df = pd.DataFrame(
                [{"낱말": w["word"], "품사": w["category"], "빈도": w["count"]}
                 for w in stats["word_freq"]]
            )
            st.dataframe(wf_df, use_container_width=True, hide_index=True)
