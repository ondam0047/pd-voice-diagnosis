"""🎯 통합 분석 — 언어 + 조음 (M4 예정)."""

import streamlit as st

st.set_page_config(page_title="통합 분석", page_icon="🎯", layout="wide")

st.title("🎯 통합 분석")
st.info(
    "통합 분석은 **M4**에서 구현됩니다.\n\n"
    "음성 업로드 → 듀얼 전사·검수 → 언어 + 조음 모든 지표 + 종합 보고서 "
    "([언어 분석] [조음 분석] [종합 코멘트] 탭)."
)
st.page_link("app.py", label="← 홈으로", icon="🏠")
