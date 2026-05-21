"""🎯 통합 분석 — 언어 + 조음 (M4). 결과는 탭으로 분리 + 종합 코멘트(M5)."""

import os
import sys

import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.articulation import analyze_articulation  # noqa: E402
from modules.insights import (  # noqa: E402
    generate_insight,
    summarize_articulation,
    summarize_language,
)
from modules.shared_ui import (  # noqa: E402
    api_key_input,
    child_pairs,
    child_targets,
    get_analyzer,
    render_articulation_results,
    render_language_results,
    require_password,
    voice_dual_review,
)
from modules.transcription import TranscriptionError  # noqa: E402

st.set_page_config(page_title="통합 분석", page_icon="🎯", layout="wide")
require_password()
api_key = api_key_input()

st.title("🎯 통합 분석")
st.caption("음성 → 듀얼 전사·검수 → 언어 + 조음 분석 모두 실행 → 종합 보고서")

edited = voice_dual_review("integ", api_key)

if edited is not None and st.button("📊 통합 분석 실행", type="primary"):
    targets = child_targets(edited)
    pairs = child_pairs(edited)
    st.session_state["integ_lang"] = get_analyzer().analyze(targets) if targets else None
    st.session_state["integ_artic"] = analyze_articulation(pairs) if pairs else None
    st.session_state.pop("integ_insight", None)
    if not targets:
        st.warning("아동 목표어 발화가 없습니다.")

lang = st.session_state.get("integ_lang")
artic = st.session_state.get("integ_artic")

if lang is not None or artic is not None:
    st.divider()
    tab_lang, tab_artic, tab_report = st.tabs(["언어 분석", "조음 분석", "종합 코멘트"])

    with tab_lang:
        if lang is not None:
            render_language_results(lang)
        else:
            st.info("아동 목표어 발화가 없어 언어 분석을 건너뛰었습니다.")

    with tab_artic:
        if artic is not None:
            render_articulation_results(artic)
        else:
            st.info("목표어·산출형이 모두 입력된 아동 발화가 없어 조음 분석을 건너뛰었습니다.")

    with tab_report:
        st.subheader("자동 요약 (사실)")
        if lang is not None:
            st.text(summarize_language(lang))
        if artic is not None:
            st.text(summarize_articulation(artic))

        st.subheader("LLM 종합 코멘트")
        if st.button("🧠 종합 코멘트 생성"):
            try:
                with st.spinner("코멘트 생성 중…"):
                    st.session_state["integ_insight"] = generate_insight(
                        articulation=artic, language=lang, api_key=api_key)
            except TranscriptionError as e:
                st.error(str(e))
        if st.session_state.get("integ_insight"):
            st.markdown(st.session_state["integ_insight"])

st.divider()
st.page_link("app.py", label="← 홈으로", icon="🏠")
