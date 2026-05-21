"""자발화 분석 도구 — 메인 (홈 + 모드 선택)."""

import os
import sys

import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from modules.shared_ui import api_key_input, require_password  # noqa: E402

st.set_page_config(
    page_title="자발화 분석 도구",
    page_icon="🎙️",
    layout="wide",
)
require_password()
api_key_input()

st.title("🎙️ 자발화 분석 도구")
st.caption("언어치료 자발화 분석 — 음성/텍스트 → 형태소·조음 분석 → 보고서 (로컬 도구)")

st.markdown("### 어떤 분석을 하시겠어요?")

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    with st.container(border=True):
        st.subheader("📝 언어 분석")
        st.write("**MLU, TTR, NDW**")
        st.write("문법형태소 분포")
        st.write("입력: 텍스트 또는 음성")
        st.page_link("pages/1_📝_언어_분석.py", label="시작하기 →", use_container_width=True)

with col2:
    with st.container(border=True):
        st.subheader("🔊 조음 분석")
        st.write("**PCC, 음운 오류**")
        st.write("컨퓨전 매트릭스 · 위치별 오류")
        st.write("입력: 음성 필수")
        st.page_link("pages/2_🔊_조음_분석.py", label="시작하기 →", use_container_width=True)

with col3:
    with st.container(border=True):
        st.subheader("🎯 통합 분석")
        st.write("**언어 + 조음**")
        st.write("종합 보고서")
        st.write("입력: 음성 필수")
        st.page_link("pages/3_🎯_통합_분석.py", label="시작하기 →", use_container_width=True)

st.divider()

with st.expander("ℹ️ 분석 모드 안내"):
    st.markdown(
        """
        - **📝 언어 분석만** — 텍스트 직접 입력 또는 음성 업로드(Whisper 자동 전사 → 임상가 검수).
          형태소 분석으로 MLU-w, MLU-m, TTR, NDW, TNW, 문법형태소 분포 산출.
        - **🔊 조음 분석만** — 음성 업로드 → 듀얼 전사(Whisper 표준어 + GPT-4o audio 산출형) →
          임상가 듀얼 검수 → 컨퓨전 매트릭스, PCC, 음소별 정확도, 위치별 오류.
        - **🎯 통합 분석** — 음성 업로드 → 듀얼 전사·검수 → 언어 + 조음 모든 지표 + 종합 보고서.

        > 환자 음성은 로컬에만 저장됩니다. OpenAI API 키는 `.env`에 설정하세요.
        """
    )

st.caption("M1: 언어 분석(텍스트 입력)까지 구현됨. 음성/조음/통합은 후속 마일스톤(M2~M4).")
