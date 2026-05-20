"""🔊 조음 분석 — PCC/오류 패턴 (M3 예정)."""

import streamlit as st

st.set_page_config(page_title="조음 분석", page_icon="🔊", layout="wide")

st.title("🔊 조음 분석")
st.info(
    "조음 분석은 **M3**에서 구현됩니다.\n\n"
    "음성 업로드 → 듀얼 전사(Whisper 표준어 + GPT-4o audio 산출형) → 임상가 듀얼 검수 → "
    "컨퓨전 매트릭스 · PCC · 음소별 정확도 · 위치별 오류."
)
st.page_link("app.py", label="← 홈으로", icon="🏠")
