"""🔊 조음 분석 — 듀얼 전사(목표어/산출형) → 컨퓨전 매트릭스 · PCC · 위치별 오류 (M3)."""

import json
import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.articulation import POSITION_ORDER, analyze_articulation  # noqa: E402
from modules.transcription import (  # noqa: E402
    TranscriptionError,
    format_ts,
    slice_audio,
    transcribe_produced,
    transcribe_target,
)

st.set_page_config(page_title="조음 분석", page_icon="🔊", layout="wide")

FEW_SHOT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "few_shot_examples.json",
)


@st.cache_data
def load_few_shot() -> dict:
    with open(FEW_SHOT_PATH, encoding="utf-8") as f:
        return json.load(f)


def show_articulation(result: dict) -> None:
    s = result["summary"]
    st.subheader("결과")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PCC (자음정확도)", f"{result['pcc']}%")
    c2.metric("목표 자음 수", s["total_consonants"])
    c3.metric("오류 수", s["error_count"])
    c4.metric("첨가", s["additions"])
    st.caption("PCC = 정확 자음 / 목표 자음 × 100. 목표어는 g2p 발음형으로 변환 후 비교 (초성 ㅇ 제외).")

    st.divider()

    # 컨퓨전 매트릭스
    st.subheader("컨퓨전 매트릭스 (목표 → 산출)")
    cm = result["confusion_matrix"]
    if cm:
        targets = sorted(cm.keys())
        produced = sorted({p for row in cm.values() for p in row})
        z = [[cm[t].get(p, 0) for p in produced] for t in targets]
        fig = px.imshow(
            z, x=produced, y=targets, text_auto=True, aspect="auto",
            color_continuous_scale="Reds",
            labels=dict(x="산출 음소", y="목표 음소", color="빈도"),
        )
        fig.update_layout(height=max(320, 40 * len(targets)))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("∅ = 생략(대응 산출 음소 없음)")
    else:
        st.success("자음 오류가 없습니다.")

    st.divider()

    # 위치별 오류
    st.subheader("위치별 오류")
    pe, pt = result["position_errors"], result["position_total"]
    pos_df = pd.DataFrame({
        "위치": POSITION_ORDER,
        "오류": [pe[p] for p in POSITION_ORDER],
        "전체": [pt[p] for p in POSITION_ORDER],
    })
    pos_df["오류율(%)"] = [
        round(e / t * 100, 1) if t else 0.0
        for e, t in zip(pos_df["오류"], pos_df["전체"])
    ]
    pfig = px.bar(pos_df, x="위치", y="오류", text="오류",
                  category_orders={"위치": POSITION_ORDER},
                  color="위치", color_discrete_sequence=px.colors.qualitative.Set2)
    pfig.update_layout(xaxis_title="", showlegend=False, height=320)
    st.plotly_chart(pfig, use_container_width=True)
    st.dataframe(pos_df, use_container_width=True, hide_index=True)

    st.divider()

    # 음소별 정확도
    st.subheader("음소별 정확도")
    pa = result["phoneme_accuracy"]
    if pa:
        pa_df = pd.DataFrame({"음소": list(pa.keys()), "정확도(%)": list(pa.values())})
        afig = px.bar(pa_df, x="음소", y="정확도(%)", text="정확도(%)",
                      color="정확도(%)", color_continuous_scale="RdYlGn", range_y=[0, 100])
        afig.update_layout(xaxis_title="", coloraxis_showscale=False, height=340)
        st.plotly_chart(afig, use_container_width=True)

    # 오류 상세
    if result["errors"]:
        with st.expander(f"오류 상세 ({len(result['errors'])}건)"):
            st.dataframe(
                pd.DataFrame(result["errors"]).rename(columns={
                    "target": "목표", "produced": "산출", "position": "위치", "word": "어절"}),
                use_container_width=True, hide_index=True,
            )


# ===================== 페이지 본문 =====================
st.title("🔊 조음 분석")
st.caption("음성 → 듀얼 전사(Whisper 목표어 + GPT-4o audio 산출형) → 임상가 검수 → PCC·컨퓨전 매트릭스")

uploaded = st.file_uploader("음성 파일 (.mp3, .wav, .m4a)", type=["mp3", "wav", "m4a"])

if uploaded is not None:
    st.audio(uploaded)
    if st.button("🎙️ 목표어 전사 시작 (Whisper)", type="primary"):
        try:
            with st.spinner("Whisper 전사 중…"):
                segs = transcribe_target(uploaded.name, uploaded.getvalue())
            st.session_state["artic_segments"] = segs
            st.session_state["artic_audio"] = (uploaded.name, uploaded.getvalue())
            st.session_state["produced_map"] = {}
            st.success(f"{len(segs)}개 발화 전사 완료. 화자 지정 후 산출형을 입력/생성하세요.")
        except TranscriptionError as e:
            st.error(str(e))

segs = st.session_state.get("artic_segments")
if segs:
    pmap = st.session_state.get("produced_map", {})
    st.markdown("**듀얼 검수** — 화자(아동/치료사/제외) 지정 · 목표어/산출형 수정")
    base_df = pd.DataFrame([
        {"#": s["index"],
         "시간": f"{format_ts(s['start'])}–{format_ts(s['end'])}",
         "화자": "아동",
         "목표어": s["text"],
         "산출형": pmap.get(s["index"], "")}
        for s in segs
    ])
    edited = st.data_editor(
        base_df, key="artic_table", use_container_width=True, hide_index=True,
        disabled=["#", "시간"],
        column_config={
            "화자": st.column_config.SelectboxColumn(
                "화자", options=["아동", "치료사", "제외"], required=True, width="small"),
            "목표어": st.column_config.TextColumn("목표어 (표준어)", width="medium"),
            "산출형": st.column_config.TextColumn("산출형 (실제 발음)", width="medium"),
        },
    )

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🤖 GPT-4o로 산출형 자동 생성 (아동·빈칸만)"):
            fname, abytes = st.session_state["artic_audio"]
            few = load_few_shot()
            new_map = dict(pmap)
            done, failed = 0, 0
            seg_by_idx = {s["index"]: s for s in segs}
            child_empty = edited[(edited["화자"] == "아동") & (edited["산출형"].fillna("") == "")]
            with st.spinner(f"산출형 생성 중… ({len(child_empty)}건)"):
                for idx in child_empty["#"].tolist():
                    s = seg_by_idx[idx]
                    try:
                        clip, _ = slice_audio(abytes, fname, s["start"], s["end"])
                        new_map[idx] = transcribe_produced("seg.wav", clip, few)
                        done += 1
                    except TranscriptionError:
                        failed += 1
            st.session_state["produced_map"] = new_map
            if done:
                st.success(f"{done}건 생성 완료.")
            if failed:
                st.warning(f"{failed}건 실패 (wav 권장 · API 키/네트워크 확인).")
            st.rerun()
        st.caption("자동 생성은 빈 산출형(아동)만 채웁니다. 입력한 산출형은 보존됩니다.")

    with col_b:
        run = st.button("📊 분석 실행", type="primary")

    if run:
        rows = edited[edited["화자"] == "아동"]
        pairs = [
            (str(t).strip(), str(p).strip())
            for t, p in zip(rows["목표어"], rows["산출형"])
            if str(t).strip() and str(p).strip()
        ]
        if not pairs:
            st.warning("목표어·산출형이 모두 입력된 아동 발화가 없습니다.")
        else:
            st.divider()
            show_articulation(analyze_articulation(pairs))
else:
    st.info("음성을 업로드하고 목표어 전사를 시작하세요. (산출형은 GPT-4o 자동 생성 또는 직접 입력)")

st.divider()
st.page_link("app.py", label="← 홈으로", icon="🏠")
