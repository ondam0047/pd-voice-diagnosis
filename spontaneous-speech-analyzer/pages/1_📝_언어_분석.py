"""📝 언어 분석 — MLU/TTR/NDW (M1: 텍스트 입력 모드).

낱말 단위는 품사 기준(체언 + 용언).
"""

import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

# 프로젝트 루트를 import 경로에 추가 (페이지 직접 실행 대비)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.morpheme import MorphemeAnalyzer  # noqa: E402

st.set_page_config(page_title="언어 분석", page_icon="📝", layout="wide")

# 엑셀 Sheet1 아동 연결발화 (산출형 그대로) — 테스트/예시용
SAMPLE_UTTERANCES = """동글동글 돌면 되요. 우리십이 시골이라서 눈샤람 만들어팠는데 갑자기 만녹고 막 넘어시고
네, 그냥 부셔져요
모르겠어요. 많이 가봐서.
우리 가속, 우리 가속이랑 다치 가소고, 그다음에는 킁삼손네 가속, 작은 삼손네 가속
춘치 마이 생겼고 어린 이가 빠져나오라고 하니까 뽑았고
용기를 내어서 울지 않았으니까 우리 엄마는 잘했다 그래서 맛있는 것도 많이 사쉈고"""


@st.cache_resource(show_spinner="형태소 분석기 로딩 중…")
def get_analyzer() -> MorphemeAnalyzer:
    return MorphemeAnalyzer()


st.title("📝 언어 분석")
st.caption("MLU-w · MLU-m · TTR · NDW · TNW · 문법형태소 분포  ·  낱말 단위 = 체언 + 용언")

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
    placeholder="예)\n동글동글 돌면 되요\n우리 가속이랑 다치 가소고",
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
    c2.metric("MLU-w (평균 낱말)", stats["mlu_w"])
    c3.metric("MLU-m (평균 형태소)", stats["mlu_m"])
    c4.metric("TTR (어휘 다양도)", stats["ttr"])

    c5, c6, c7 = st.columns(3)
    c5.metric("TNW (총 낱말 수)", stats["tnw"])
    c6.metric("NDW (서로 다른 낱말)", stats["ndw"])
    c7.metric("총 형태소 수", stats["total_morphemes"])

    st.caption(
        "낱말 = 체언 + 용언 (품사 기준, 세종 태그셋). "
        "MLU-w = 총 낱말 / 발화 수 · MLU-m = 총 형태소 / 발화 수 · TTR = NDW / TNW."
    )

    st.divider()

    # --- 낱말: 체언 / 용언 분포 ---
    st.subheader("낱말 분포 (체언 / 용언)")
    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("체언 (낱말)", stats["cheeon_count"])
    cc2.metric("용언 (낱말)", stats["yongeon_count"])
    cc3.metric("체언 (서로 다름)", stats["cheeon_ndw"])
    cc4.metric("용언 (서로 다름)", stats["yongeon_ndw"])

    wc_df = pd.DataFrame({
        "품사": ["체언", "용언"],
        "총 낱말 수": [stats["cheeon_count"], stats["yongeon_count"]],
        "서로 다른 낱말": [stats["cheeon_ndw"], stats["yongeon_ndw"]],
    })
    fig_wc = px.bar(
        wc_df.melt(id_vars="품사", var_name="구분", value_name="빈도"),
        x="품사", y="빈도", color="구분", barmode="group", text="빈도",
        color_discrete_map={"총 낱말 수": "#4C78A8", "서로 다른 낱말": "#9ECAE1"},
    )
    fig_wc.update_layout(xaxis_title="", height=360, legend_title="")
    st.plotly_chart(fig_wc, use_container_width=True)

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
            {"#": i + 1, "발화": u["text"], "낱말": u["words"],
             "체언": u["cheeon"], "용언": u["yongeon"], "형태소": u["morphemes"]}
            for i, u in enumerate(result["utterances"])
        ]
    )
    st.dataframe(detail_df, use_container_width=True, hide_index=True)

    with st.expander("형태소 분해 보기 (체언/용언 표시)"):
        for i, u in enumerate(result["utterances"]):
            st.markdown(f"**{i + 1}. {u['text']}**")
            parts = []
            for t in u["tokens"]:
                mark = f" ⟨{t['word_class']}⟩" if t["word_class"] else ""
                parts.append(f"`{t['form']}`/{t['tag']}{mark}")
            st.markdown("  ".join(parts))

    # --- 낱말 빈도 ---
    if stats["word_freq"]:
        with st.expander("낱말 빈도 (체언+용언, 상위 20)"):
            wf_df = pd.DataFrame(
                [{"낱말": w["word"], "품사": w["word_class"], "빈도": w["count"]}
                 for w in stats["word_freq"]]
            )
            st.dataframe(wf_df, use_container_width=True, hide_index=True)
