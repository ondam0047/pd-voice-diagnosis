"""📝 언어 분석 — MLU/TTR/NDW (M1: 텍스트 입력 모드)."""

import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

# 프로젝트 루트를 import 경로에 추가 (페이지 직접 실행 대비)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.morpheme import MorphemeAnalyzer  # noqa: E402

st.set_page_config(page_title="언어 분석", page_icon="📝", layout="wide")

# 엑셀 Sheet1 아동 연결발화에서 추출한 표준어 형태 (테스트/예시용)
SAMPLE_UTTERANCES = """우리 집이 시골이라서 눈사람 만들기 힘들어서
동글동글 돌면 돼요
만들어봤는데 갑자기 막 녹고 부서져요
넘어지고 다쳤어요
우리 가족 같이 갔고 큰삼촌네 초등학교에 갔어요
충치 때문에 치과에 갔어요
여행 가서 용기를 내어서 놀이기구 탔어요
삼촌이 장난감 사주셨고 너무 좋았어요"""


@st.cache_resource(show_spinner="형태소 분석기 로딩 중…")
def get_analyzer() -> MorphemeAnalyzer:
    return MorphemeAnalyzer()


st.title("📝 언어 분석")
st.caption("MLU-w · MLU-m · TTR · NDW · TNW · 문법형태소 분포")

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

if st.button("예시 발화 불러오기", help="엑셀 아동 연결발화 기반 예시"):
    st.session_state["lang_text"] = SAMPLE_UTTERANCES

text = st.text_area(
    "발화 입력",
    key="lang_text",
    height=220,
    placeholder="예)\n우리 집이 시골이라서 눈사람 만들기 힘들어서\n동글동글 돌면 돼요",
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

    # --- 핵심 지표 카드 ---
    st.subheader("핵심 지표")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("발화 수", stats["utterance_count"])
    c2.metric("MLU-w (평균 어절)", stats["mlu_w"])
    c3.metric("MLU-m (평균 형태소)", stats["mlu_m"])
    c4.metric("TTR (어휘 다양도)", stats["ttr"])

    c5, c6, c7 = st.columns(3)
    c5.metric("TNW (총 어절 수)", stats["tnw"])
    c6.metric("NDW (서로 다른 어절)", stats["ndw"])
    c7.metric("총 형태소 수", stats["total_morphemes"])

    st.caption(
        "MLU-w = 총 어절 수 / 발화 수 · MLU-m = 총 형태소 수 / 발화 수 · "
        "TTR = NDW / TNW (어절 기준). 낱말 단위는 공백 기준 어절을 사용합니다."
    )

    st.divider()

    # --- 문법형태소 분포 ---
    st.subheader("문법형태소 분포")
    gram = stats["grammatical_morphemes"]
    if gram:
        gram_df = pd.DataFrame(
            {"문법형태소": list(gram.keys()), "빈도": list(gram.values())}
        )
        fig = px.bar(
            gram_df, x="문법형태소", y="빈도", text="빈도",
            color="빈도", color_continuous_scale="Blues",
        )
        fig.update_layout(xaxis_title="", coloraxis_showscale=False, height=380)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("문법형태소(조사·어미·접사)가 검출되지 않았습니다.")

    # --- 품사 대분류 분포 ---
    pos = stats["pos_distribution"]
    if pos:
        with st.expander("품사 대분류 분포"):
            pos_df = pd.DataFrame(
                {"품사": list(pos.keys()), "빈도": list(pos.values())}
            )
            st.plotly_chart(
                px.bar(pos_df, x="품사", y="빈도", text="빈도").update_layout(
                    xaxis_title="", height=320
                ),
                use_container_width=True,
            )

    st.divider()

    # --- 발화별 상세 ---
    st.subheader("발화별 상세")
    detail_df = pd.DataFrame(
        [
            {"#": i + 1, "발화": u["text"], "어절": u["words"], "형태소": u["morphemes"]}
            for i, u in enumerate(result["utterances"])
        ]
    )
    st.dataframe(detail_df, use_container_width=True, hide_index=True)

    with st.expander("형태소 분해 보기"):
        for i, u in enumerate(result["utterances"]):
            st.markdown(f"**{i + 1}. {u['text']}**")
            morph_str = "  ".join(
                f"`{t['form']}`/{t['tag']}" for t in u["tokens"]
            )
            st.markdown(morph_str)

    # --- 어절 빈도 ---
    if stats["word_freq"]:
        with st.expander("어절 빈도 (상위 20)"):
            wf_df = pd.DataFrame(stats["word_freq"], columns=["어절", "빈도"])
            st.dataframe(wf_df, use_container_width=True, hide_index=True)
