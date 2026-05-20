"""📝 언어 분석 — MLU/TTR/NDW.

M1: 텍스트 입력. M2: 음성 업로드 + Whisper 표준어 전사 → 임상가 검수 → 분석.
낱말 = 체언+용언+수식언+독립언. 의미/문법 영역 분석.
"""

import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

# 프로젝트 루트를 import 경로에 추가 (페이지 직접 실행 대비)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.morpheme import (  # noqa: E402
    BROAD_OF,
    GRAM_ORDER,
    SEMANTIC_ORDER,
    SENTENCE_TYPES,
    MorphemeAnalyzer,
)
from modules.transcription import (  # noqa: E402
    TranscriptionError,
    format_ts,
    transcribe_target,
)

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


def show_results(result: dict) -> None:
    """분석 결과 렌더링."""
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

    # --- 문법형태소 세분류 ---
    st.subheader("문법형태소 (세분류)")
    gcat = stats["gram_categories"]
    gcat_df = pd.DataFrame(
        {"문법형태소": GRAM_ORDER, "빈도": [gcat[c] for c in GRAM_ORDER]}
    )
    gfig = px.bar(
        gcat_df, x="문법형태소", y="빈도", text="빈도",
        category_orders={"문법형태소": GRAM_ORDER},
        color="빈도", color_continuous_scale="Blues",
    )
    gfig.update_layout(xaxis_title="", coloraxis_showscale=False, height=340)
    st.plotly_chart(gfig, use_container_width=True)
    st.caption("피동·사동 접사는 형태소 분석기가 어간에 병합하여 자동 분리되지 않습니다 (임상가 검수 항목).")

    with st.expander("문법형태소 상세 (조사·어미 종류별)"):
        gram = stats["grammatical_morphemes"]
        if gram:
            st.dataframe(
                pd.DataFrame({"종류": list(gram.keys()), "빈도": list(gram.values())}),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("검출된 문법형태소가 없습니다.")

    st.divider()

    # --- 문장유형 (자동 추정) ---
    st.subheader("문장유형 (자동 추정)")
    sent = stats["sentence_types"]
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("단문", sent["단문"])
    sc2.metric("이어진문장", sent["이어진문장"])
    sc3.metric("안긴문장", sent["안긴문장"])
    sent_df = pd.DataFrame(
        {"문장유형": SENTENCE_TYPES, "발화 수": [sent[s] for s in SENTENCE_TYPES]}
    )
    sfig = px.bar(
        sent_df, x="문장유형", y="발화 수", text="발화 수",
        category_orders={"문장유형": SENTENCE_TYPES},
        color="문장유형",
        color_discrete_map={"단문": "#4C78A8", "이어진문장": "#F58518", "안긴문장": "#54A24B"},
    )
    sfig.update_layout(xaxis_title="", showlegend=False, height=320)
    st.plotly_chart(sfig, use_container_width=True)
    st.caption(
        "연결어미(이어진문장)·전성어미/관형절(안긴문장) 기반 자동 추정입니다. "
        "관형 수식·인용절 등은 정확도가 낮아 임상가 검수가 필요합니다."
    )

    st.divider()

    # --- 발화별 상세 ---
    st.subheader("발화별 상세")
    detail_df = pd.DataFrame(
        [
            {"#": i + 1, "발화": u["text"], "문장유형": u["sentence_type"],
             "낱말": u["words"], "형태소": u["morphemes"],
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


# ===================== 페이지 본문 =====================
st.title("📝 언어 분석")
st.caption("MLU-w · MLU-m · TTR · NDW · TNW  ·  낱말 = 체언 + 용언 + 수식언 + 독립언")

input_mode = st.radio("입력 방식", ["텍스트 직접 입력", "음성 업로드"], horizontal=True)

utterances: list[str] | None = None

if input_mode == "텍스트 직접 입력":
    st.markdown("**발화를 한 줄에 하나씩 입력하세요.** (한 줄 = 한 발화)")
    st.caption("정확한 지표를 위해 표준어로 정규화하고 반복·수정·간투사(마디)는 제외한 전사를 권장합니다.")
    if st.button("예시 발화 불러오기"):
        st.session_state["lang_text"] = SAMPLE_UTTERANCES
    text = st.text_area(
        "발화 입력", key="lang_text", height=220,
        placeholder="예)\n엄마랑 아빠랑 같이 큰집에 갔어요\n그러면 우리 같이 영화 보러 가요",
    )
    if st.button("분석 실행", type="primary"):
        utterances = [line for line in text.splitlines() if line.strip()]
        if not utterances:
            st.warning("발화를 한 줄 이상 입력하세요.")
            utterances = None

else:  # 음성 업로드
    st.markdown(
        "**음성을 업로드하면 Whisper로 전사합니다.** 표에서 발화별 화자(아동/치료사/제외)를 "
        "지정하면 **아동 발화만** 분석합니다."
    )
    st.caption("OpenAI API 키 필요 · Whisper 25MB 제한. 전사 후 화자 지정·표준어 수정 검수하세요.")
    uploaded = st.file_uploader("음성 파일 (.mp3, .wav, .m4a)", type=["mp3", "wav", "m4a"])

    if uploaded is not None:
        st.audio(uploaded)
        if st.button("🎙️ 자동 전사 시작", type="primary"):
            try:
                with st.spinner("Whisper 전사 중…"):
                    segs = transcribe_target(uploaded.name, uploaded.getvalue())
                st.session_state["voice_segments"] = segs
                st.success(f"{len(segs)}개 발화 전사 완료. 화자를 지정하고 검수하세요.")
            except TranscriptionError as e:
                st.error(str(e))

    segs = st.session_state.get("voice_segments")
    if segs:
        st.markdown("**발화별 검수** — 화자 지정(아동/치료사/제외) + 표준어 수정")
        base_df = pd.DataFrame([
            {"#": s["index"],
             "시간": f"{format_ts(s['start'])}–{format_ts(s['end'])}",
             "화자": "아동",
             "전사": s["text"]}
            for s in segs
        ])
        edited_df = st.data_editor(
            base_df, key="voice_table", use_container_width=True, hide_index=True,
            disabled=["#", "시간"],
            column_config={
                "화자": st.column_config.SelectboxColumn(
                    "화자", options=["아동", "치료사", "제외"], required=True, width="small"),
                "전사": st.column_config.TextColumn("전사 (수정 가능)", width="large"),
            },
        )
        n_child = int((edited_df["화자"] == "아동").sum())
        st.caption(f"아동 발화로 지정된 항목: {n_child}개 (이 항목만 분석)")
        if st.button("분석 실행", type="primary"):
            child_rows = edited_df[edited_df["화자"] == "아동"]
            utterances = [t for t in child_rows["전사"].tolist() if str(t).strip()]
            if not utterances:
                st.warning("아동 발화로 지정된 항목이 없습니다. 화자를 지정하세요.")
                utterances = None

if utterances:
    show_results(get_analyzer().analyze(utterances))
