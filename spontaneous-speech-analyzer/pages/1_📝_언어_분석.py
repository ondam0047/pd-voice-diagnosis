"""📝 언어 분석 — MLU/TTR/NDW.

텍스트 입력 또는 음성 업로드(Whisper 전사 + 화자 지정 → 아동 발화만).
낱말 = 체언+용언+수식언+독립언. 의미/문법 영역 분석.
"""

import os
import sys

import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.shared_ui import (  # noqa: E402
    api_key_input,
    get_analyzer,
    render_language_results,
    require_password,
)
from modules.transcription import (  # noqa: E402
    TranscriptionError,
    format_ts,
    transcribe_target,
)

st.set_page_config(page_title="언어 분석", page_icon="📝", layout="wide")
require_password()
api_key = api_key_input()

SAMPLE_UTTERANCES = """엄마랑 아빠랑 같이 큰집에 갔어요
토끼가 추운데 죽어서 너무 슬펐어요
시골이라서 연을 날릴 수 있어요
국이 너무 뜨거워서 못 먹었어요
그러면 우리 같이 영화 보러 가요"""

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
        placeholder="예)\n엄마랑 아빠랑 같이 큰집에 갔어요\n그러면 우리 같이 영화 보러 가요")
    if st.button("분석 실행", type="primary"):
        utterances = [line for line in text.splitlines() if line.strip()]
        if not utterances:
            st.warning("발화를 한 줄 이상 입력하세요.")
            utterances = None

else:  # 음성 업로드 (언어 분석은 목표어만 필요)
    st.markdown("**음성을 업로드하면 Whisper로 전사합니다.** 화자(아동/치료사/제외)를 지정하면 **아동 발화만** 분석합니다.")
    st.caption("OpenAI API 키 필요 · Whisper 25MB 제한.")
    uploaded = st.file_uploader("음성 파일 (.mp3, .wav, .m4a)", type=["mp3", "wav", "m4a"])
    if uploaded is not None:
        st.audio(uploaded)
        if st.button("🎙️ 자동 전사 시작", type="primary"):
            try:
                with st.spinner("Whisper 전사 중…"):
                    segs = transcribe_target(
                        uploaded.name, uploaded.getvalue(), api_key=api_key)
                st.session_state["lang_voice_segments"] = segs
                st.success(f"{len(segs)}개 발화 전사 완료. 화자를 지정하고 검수하세요.")
            except TranscriptionError as e:
                st.error(str(e))

    segs = st.session_state.get("lang_voice_segments")
    if segs:
        st.markdown("**발화별 검수** — 화자 지정 + 표준어 수정 · 행 추가/삭제로 발화 분리·병합")
        base_df = pd.DataFrame([
            {"#": s["index"], "시간": f"{format_ts(s['start'])}–{format_ts(s['end'])}",
             "화자": "아동", "전사": s["text"]}
            for s in segs
        ])
        edited_df = st.data_editor(
            base_df, key="lang_voice_table", use_container_width=True, hide_index=True,
            num_rows="dynamic", disabled=["#", "시간"],
            column_config={
                "화자": st.column_config.SelectboxColumn(
                    "화자", options=["아동", "치료사", "제외"], required=True, width="small"),
                "전사": st.column_config.TextColumn("전사 (수정 가능)", width="large"),
            })
        n_child = int((edited_df["화자"] == "아동").sum())
        st.caption(f"아동 발화로 지정된 항목: {n_child}개 (이 항목만 분석)")
        if st.button("분석 실행", type="primary"):
            child = edited_df[edited_df["화자"] == "아동"]
            utterances = [t for t in child["전사"].tolist() if str(t).strip()]
            if not utterances:
                st.warning("아동 발화로 지정된 항목이 없습니다.")
                utterances = None

if utterances:
    render_language_results(get_analyzer().analyze(utterances))
