import streamlit as st
import parselmouth
from parselmouth.praat import call
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import platform
from sklearn.ensemble import RandomForestClassifier
from scipy.signal import find_peaks

# --- 페이지 기본 설정 ---
st.set_page_config(page_title="PD 음성 변별 진단 시스템", layout="wide")

# ==========================================
# [한글 폰트 설정]
# ==========================================
def setup_korean_font():
    system_name = platform.system()
    if system_name == 'Windows':
        try:
            font_path = "C:/Windows/Fonts/malgun.ttf"
            font_name = fm.FontProperties(fname=font_path).get_name()
            plt.rc('font', family=font_name)
        except:
            plt.rc('font', family='Malgun Gothic')
    elif system_name == 'Darwin': 
        plt.rc('font', family='AppleGothic')
    else: 
        plt.rc('font', family='NanumGothic')
    plt.rcParams['axes.unicode_minus'] = False

setup_korean_font()

# ==========================================
# 0. 머신러닝 모델 학습 (VHI-10 구조 반영)
# ==========================================
@st.cache_resource
def train_models():
    DATA_FILE = "training_data.csv" # 혹은 xlsx
    df = None
    
    # 1. 데이터 로드 (CSV, Excel 지원)
    loaders = [
        (lambda f: pd.read_excel(f.replace(".csv", ".xlsx")), "excel"), # xlsx 우선 시도
        (lambda f: pd.read_csv(f, encoding='utf-8'), "utf-8"),
        (lambda f: pd.read_csv(f, encoding='cp949'), "cp949"),
        (lambda f: pd.read_csv(f, encoding='euc-kr'), "euc-kr")
    ]
    
    # 파일 확장자 체크 및 로드 시도
    base_name = "training_data"
    file_found = False
    
    for ext in [".xlsx", ".csv"]:
        if os.path.exists(base_name + ext):
            DATA_FILE = base_name + ext
            file_found = True
            break
            
    if not file_found:
        return None, None

    # 로더 실행
    for loader, enc_name in loaders:
        try:
            df_raw = loader(DATA_FILE)
            if df_raw is not None and not df_raw.empty:
                break
        except:
            continue
            
    if df_raw is not None:
        try:
            data_list = []
            for _, row in df_raw.iterrows():
                label = str(row.get('진단결과 (Label)', 'Normal')).strip()
                
                # 라벨 정규화
                if 'normal' in label.lower():
                    diagnosis = "Normal"
                    subgroup = "Normal"
                elif 'pd_intensity' in label.lower():
                    diagnosis = "Parkinson"
                    subgroup = "강도 집단"
                elif 'pd_rate' in label.lower():
                    diagnosis = "Parkinson"
                    subgroup = "말속도 집단"
                elif 'pd_articulation' in label.lower():
                    diagnosis = "Parkinson"
                    subgroup = "조음 집단"
                else:
                    continue # 알 수 없는 라벨 제외

                # [핵심 로직] VHI 데이터 전처리 및 스케일링
                # 데이터셋의 컬럼명에 따라 가져오기
                raw_total = row.get('VHI총점', 0)
                raw_p = row.get('VHI_신체', 0)
                raw_f = row.get('VHI_기능', 0)
                raw_e = row.get('VHI_정서', 0)
                
                # VHI-30 데이터(총점 > 40)인 경우 VHI-10 스케일로 변환
                # VHI-10 구조: 기능(20점), 신체(12점), 정서(8점) 만점
                if raw_total > 40: 
                    # VHI-30은 각 영역이 40점 만점이므로 비율대로 축소
                    vhi_f = (raw_f / 40.0) * 20.0
                    vhi_p = (raw_p / 40.0) * 12.0
                    vhi_e = (raw_e / 40.0) * 8.0
                    vhi_total = vhi_f + vhi_p + vhi_e
                else:
                    vhi_total = raw_total
                    vhi_f = raw_f
                    vhi_p = raw_p
                    vhi_e = raw_e
                
                # 청지각 변수 처리
                p_pitch = row.get('음도(청지각)', 0)
                p_prange = row.get('음도범위(청지각)', 0)
                p_loud = row.get('강도(청지각)', 0)
                p_rate = row.get('말속도(청지각)', 0)
                p_artic = row.get('조음정확도(청지각)', 0)
                
                data_list.append([
                    row.get('F0', 0), row.get('Range', 0), row.get('강도(dB)', 0), row.get('SPS', 0),
                    vhi_total, vhi_p, vhi_f, vhi_e,
                    p_pitch, p_prange, p_loud, p_rate, p_artic,
                    diagnosis, subgroup
                ])
            
            df = pd.DataFrame(data_list, columns=[
                'F0', 'Range', 'Intensity', 'SPS', 
                'VHI_Total', 'VHI_P', 'VHI_F', 'VHI_E', 
                'P_Pitch', 'P_Range', 'P_Loudness', 'P_Rate', 'P_Artic', 
                'Diagnosis', 'Subgroup'
            ])
            
            # 결측치 보완 (음향 변수는 평균으로)
            acoustic_vars = ['F0', 'Range', 'Intensity', 'SPS']
            for col in acoustic_vars:
                df[col] = df[col].fillna(df[col].mean())
            
        except Exception as e:
            st.error(f"데이터 전처리 오류: {e}")
            return None, None

    if df is None or df.empty:
        st.warning("⚠️ 학습 데이터가 유효하지 않습니다.")
        return None, None

    # --- 모델 학습 시작 ---
    # [Step 1] Normal vs Parkinson (Binary)
    # 정서(VHI_E) 변수가 말속도 집단 변별에 중요하므로 포함
    feats_step1 = ['F0', 'Range', 'Intensity', 'SPS', 'VHI_Total', 'VHI_P', 'VHI_F', 'VHI_E']
    model_step1 = RandomForestClassifier(n_estimators=200, random_state=42)
    model_step1.fit(df[feats_step1], df['Diagnosis'])

    # [Step 2] PD Subtype Classification
    df_pd = df[df['Diagnosis'] == 'Parkinson'].copy()
    
    # PD 데이터가 너무 적으면 학습 불가 처리
    if len(df_pd) < 2:
        return model_step1, None

    feats_step2 = feats_step1 + ['P_Pitch', 'P_Range', 'P_Loudness', 'P_Rate', 'P_Artic']
    
    # 결측치 처리
    for col in feats_step2:
        df_pd[col] = df_pd[col].fillna(df_pd[col].mean())
        
    model_step2 = RandomForestClassifier(n_estimators=200, random_state=42)
    model_step2.fit(df_pd[feats_step2], df_pd['Subgroup'])

    return model_step1, model_step2

try:
    model_step1, model_step2 = train_models()
except Exception as e:
    st.error(f"모델 학습 중 오류 발생: {e}")
    model_step1, model_step2 = None, None

# --- Sidebar ---
with st.sidebar:
    st.title("👤 대상자 정보")
    subject_name = st.text_input("이름", "대상자")
    subject_age = st.number_input("나이", 1, 120, 60)
    subject_gender = st.selectbox("성별", ["남", "여", "기타"])
    st.info("※ 본 시스템은 VHI-10 (총점 40점) 기준을 사용합니다.")

TEMP_FILENAME = "temp_for_analysis.wav"

# ==========================================
# [함수] 자동 조음 분석
# ==========================================
def auto_detect_smr_events(sound_path, top_n=10):
    try:
        sound = parselmouth.Sound(sound_path)
        intensity = sound.to_intensity(time_step=0.005)
        times = intensity.xs()
        values = intensity.values[0, :]
        inv_vals = -values
        peaks, properties = find_peaks(inv_vals, prominence=5, distance=40)
        candidates = []
        for p_idx in peaks:
            time_point = times[p_idx]
            v_int = values[p_idx]
            start_search = max(0, p_idx - 20)
            end_search = min(len(values), p_idx + 20)
            local_max = np.max(values[start_search:end_search])
            depth = local_max - v_int
            candidates.append({"time": time_point, "depth": depth})
        candidates.sort(key=lambda x: x['time'])
        return candidates[:top_n], len(candidates)
    except:
        return [], 0

# ==========================================
# [함수] 피치 컨투어 시각화
# ==========================================
def plot_pitch_contour_plotly(sound_path, f0_min, f0_max):
    try:
        sound = parselmouth.Sound(sound_path)
        pitch = call(sound, "To Pitch", 0.0, f0_min, f0_max)
        pitch_vals = np.array(pitch.selected_array['frequency'], dtype=np.float64)
        duration = sound.get_total_duration()
        times = np.linspace(0, duration, len(pitch_vals))
        
        valid_idx = pitch_vals != 0
        valid_p = pitch_vals[valid_idx]
        valid_t = times[valid_idx]

        if len(valid_p) > 0:
            mean_f0 = np.mean(valid_p)
            rng = np.max(valid_p) - np.min(valid_p)
        else:
            mean_f0, rng = 0, 0

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=valid_t, y=valid_p, mode='markers', marker=dict(size=4, color='red'), name='Pitch'))
        
        fig.update_layout(title="음도 컨투어", xaxis_title="Time(s)", yaxis_title="Hz", height=300, yaxis=dict(range=[0, 350]))
        return fig, mean_f0, rng, duration
    except:
        return None, 0, 0, 0

# --- UI Title ---
st.title("🧠 파킨슨병(PD) 음성 하위유형 변별 진단 시스템")
st.markdown("청지각 + 음향 + 자가보고(VHI-10) 통합 하이브리드 진단 모델")

# ==========================================
# 1. 문단 낭독 및 음성 분석
# ==========================================
st.header("1. 문단 낭독 및 음성 분석")

col_rec, col_up = st.columns(2)
if 'user_syllables' not in st.session_state: st.session_state.user_syllables = 80
if 'source_type' not in st.session_state: st.session_state.source_type = None

# [좌측: 마이크 녹음]
with col_rec:
    st.markdown("#### 🎙️ 마이크 녹음 & 문단")
    font_size = st.slider("글자 크기", 20, 50, 28)
    
    with st.expander("🔎 [2] 바닷가의 추억 (SMR/조음 정밀 진단용)", expanded=True):
        seaside_text = """
        <div style="font-size: {}px; line-height: 1.8; border: 1px solid #ddd; padding: 15px;">
        <strong>바닷가</strong>에 <strong>파도가</strong> 칩니다.<br>
        <strong>무지개</strong> 아래 <strong>바둑이</strong>가 뜁니다.<br>
        <strong>보트가</strong> 지나가고 <strong>버터구이</strong>를 먹습니다.<br>
        <strong>포토카드</strong>를 <strong>부탁해</strong>서 <strong>돋보기</strong>로 봅니다.<br>
        시장에서 <strong>빈대떡</strong>을 사 먹었습니다.
        </div>
        """.format(font_size)
        st.markdown(seaside_text, unsafe_allow_html=True)

    syllables_rec = st.number_input("전체 음절 수", 1, 500, 80)
    st.session_state.user_syllables = syllables_rec
    
    audio_buf = st.audio_input("낭독 녹음")
    if audio_buf:
        with open(TEMP_FILENAME, "wb") as f: f.write(audio_buf.read())
        st.session_state.current_wav_path = os.path.join(os.getcwd(), TEMP_FILENAME)
        st.session_state.source_type = "mic"
        st.success("녹음 완료")

# [우측: 파일 업로드]
with col_up:
    st.markdown("#### 📂 파일 업로드")
    up_file = st.file_uploader("WAV 파일 선택", type=["wav"])
    if up_file:
        with open(TEMP_FILENAME, "wb") as f: f.write(up_file.read())
        st.session_state.current_wav_path = os.path.join(os.getcwd(), TEMP_FILENAME)
        st.session_state.source_type = "upload"
        st.success("파일 준비됨")

# 분석 버튼
if st.button("🛠️ 음성 분석 실행", key="btn_anal_main"):
    if 'current_wav_path' in st.session_state:
        try:
            fig, f0, rng, dur = plot_pitch_contour_plotly(st.session_state.current_wav_path, 75, 300)
            sound = parselmouth.Sound(st.session_state.current_wav_path)
            intensity = sound.to_intensity()
            mean_db = call(intensity, "Get mean", 0, 0, "energy")
            sps = st.session_state.user_syllables / dur if dur > 0 else 0
            
            smr_events, _ = auto_detect_smr_events(st.session_state.current_wav_path)
            
            st.session_state.update({
                'f0_mean': f0, 'pitch_range': rng, 'mean_db': mean_db, 
                'sps': sps, 'duration': dur, 'fig_plotly': fig, 
                'is_analyzed': True, 'smr_events': smr_events
            })
        except Exception as e:
            st.error(f"분석 오류: {e}")
    else:
        st.warning("먼저 녹음을 하거나 파일을 업로드하세요.")

# ==========================================
# 2. 결과 및 보정
# ==========================================
if st.session_state.get('is_analyzed'):
    st.markdown("---")
    st.subheader("2. 분석 결과 및 보정")
    c1, c2 = st.columns([2, 1])
    with c1: st.plotly_chart(st.session_state['fig_plotly'], use_container_width=True)
    with c2:
        db_adj = st.slider("강도(dB) 보정", -50.0, 50.0, -10.0, help="마이크 측정 시 실제보다 크게 나올 수 있어 보정합니다.")
        final_db = st.session_state['mean_db'] + db_adj
        
        # 말속도 구간 재설정
        s_time, e_time = st.slider("말속도 분석 구간", 0.0, st.session_state['duration'], (0.0, st.session_state['duration']))
        sel_dur = max(0.1, e_time - s_time)
        final_sps = st.session_state.user_syllables / sel_dur
        
        st.metric("보정된 강도", f"{final_db:.1f} dB")
        st.metric("보정된 말속도", f"{final_sps:.2f} SPS")

    # ==========================================
    # 3. 청지각/자가보고 (VHI-10) - 정밀 매핑
    # ==========================================
    st.markdown("---")
    st.subheader("3. 청지각 평가 및 자가보고 (VHI-10)")
    
    cc1, cc2 = st.columns([1, 1.2])
    
    with cc1:
        st.markdown("#### 🔊 청지각 평가 (Clinician)")
        p_artic = st.slider("조음 정확도 (Articulation)", 0, 100, 50)
        p_pitch = st.slider("음도 (Pitch)", 0, 100, 50)
        p_prange = st.slider("음도 범위 (Pitch Range)", 0, 100, 50)
        p_loud = st.slider("강도 (Loudness)", 0, 100, 50)
        p_rate = st.slider("말속도 (Rate)", 0, 100, 50)
        
    with cc2:
        st.markdown("#### 📝 VHI-10 자가보고 (Patient)")
        st.caption("0: 전혀, 1: 거의X, 2: 가끔, 3: 자주, 4: 항상")
        
        vhi_opts = [0, 1, 2, 3, 4]
        
        with st.expander("VHI-10 문항 입력 (클릭)", expanded=True):
            # 기능(Functional, F) - 5문항
            st.markdown("**[기능적 영역 (5문항)]**")
            q1 = st.select_slider("1. (기능) 상대방이 내 말을 알아듣기 힘들어한다", options=vhi_opts)
            q2 = st.select_slider("2. (기능) 시끄러운 곳에서 이해하기 어려워한다", options=vhi_opts)
            q5 = st.select_slider("5. (기능) 음성문제로 생활에 제한을 받는다", options=vhi_opts)
            q7 = st.select_slider("7. (기능) 대화에 끼지 못해 소외감을 느낀다", options=vhi_opts)
            q8 = st.select_slider("8. (기능) 음성 문제로 수입 감소가 생긴다", options=vhi_opts)
            
            # 신체(Physical, P) - 3문항
            st.markdown("**[신체적 영역 (3문항)]**")
            q3 = st.select_slider("3. (신체) 사람들이 목소리가 왜 그러냐고 묻는다", options=vhi_opts)
            q4 = st.select_slider("4. (신체) 목소리를 내려면 힘을 주어야 한다", options=vhi_opts)
            q6 = st.select_slider("6. (신체) 목소리가 언제 맑게 나올지 알 수 없다", options=vhi_opts)
            
            # 정서(Emotional, E) - 2문항 (핵심 변수)
            st.markdown("**[정서적 영역 (2문항)]** - 말속도 유형 판별 중요 지표")
            q9 = st.select_slider("9. (정서) 내 목소리 문제로 속이 상한다", options=vhi_opts)
            q10 = st.select_slider("10. (정서) 음성 문제가 장애로 여겨진다", options=vhi_opts)

        # 영역별 계산
        vhi_f = q1 + q2 + q5 + q7 + q8 # Max 20
        vhi_p = q3 + q4 + q6           # Max 12
        vhi_e = q9 + q10               # Max 8
        vhi_total = vhi_f + vhi_p + vhi_e
        
        st.info(f"📊 VHI 결과: 총점 {vhi_total}/40 (기능 {vhi_f}/20, 신체 {vhi_p}/12, 정서 {vhi_e}/8)")

    # ==========================================
    # 4. 최종 진단 (Hybrid Logic: ML + Rules)
    # ==========================================
    st.markdown("---")
    st.subheader("4. 최종 종합 진단")
    
    if st.button("🚀 진단 결과 확인", key="btn_diag"):
        if model_step1 is None:
            st.error("모델이 로드되지 않았습니다. 학습 데이터를 확인하세요.")
        else:
            # [Step 0] Rule-based Pre-check
            if p_artic >= 78 and vhi_total < 10:
                st.success("🟢 **정상 음성 (Normal)** 범위입니다.")
                st.write("청지각적 조음 정확도가 높고, 자가 불편함(VHI)이 낮습니다.")
            else:
                # [Step 1] AI Binary Classification
                input_step1 = pd.DataFrame([[
                    st.session_state['f0_mean'], st.session_state['pitch_range'], final_db, final_sps,
                    vhi_total, vhi_p, vhi_f, vhi_e
                ]], columns=['F0', 'Range', 'Intensity', 'SPS', 'VHI_Total', 'VHI_P', 'VHI_F', 'VHI_E'])
                
                pred_1 = model_step1.predict(input_step1)[0]
                
                if pred_1 == 'Normal':
                    st.success("🟢 **정상 음성 (Normal)** 범위입니다.")
                    st.info("AI 분석 결과, 정상 데이터 패턴과 유사합니다.")
                else:
                    st.error("🔴 **파킨슨병(PD) 음성 특성**이 감지되었습니다.")
                    
                    # [Step 2] AI Subtype Classification
                    if model_step2:
                        input_step2 = pd.DataFrame([[
                            st.session_state['f0_mean'], st.session_state['pitch_range'], final_db, final_sps,
                            vhi_total, vhi_p, vhi_f, vhi_e,
                            p_pitch, p_prange, p_loud, p_rate, p_artic
                        ]], columns=feats_step2)
                        
                        pred_subtype = model_step2.predict(input_step2)[0]
                        probs = model_step2.predict_proba(input_step2)[0]
                        
                        # --- [Hybrid Logic] 가중치 기반 최종 판단 보정 ---
                        # 데이터 분석 결과: 정서 점수 비율이 높으면 '말속도 집단'일 확률이 매우 높음
                        emotional_ratio = vhi_e / 8.0
                        predicted_final = pred_subtype
                        
                        hybrid_msg = ""
                        
                        if emotional_ratio >= 0.6: # 정서 점수가 5점 이상(8점 만점)
                            hybrid_msg += "⚠️ **주의:** 높은 정서적 스트레스(VHI-정서)가 감지되었습니다. 이는 '말속도(Rate)' 유형에서 흔히 나타납니다.\n"
                            if "말속도" not in pred_subtype and final_sps > 4.5:
                                predicted_final = "말속도 집단 (재조정됨)"
                                hybrid_msg += "👉 AI 예측을 **말속도 집단**으로 보정했습니다.\n"
                        
                        if final_db < 60.0:
                            hybrid_msg += "⚠️ **참고:** 음성 강도가 60dB 미만입니다. 이는 '강도(Intensity)' 유형의 강력한 특징입니다.\n"
                            
                        if vhi_total < 15 and p_artic < 60:
                             hybrid_msg += "⚠️ **참고:** 환자의 주관적 불편함(VHI)은 낮으나 객관적 조음 정확도가 낮습니다. '조음(Articulation)' 유형의 특징일 수 있습니다.\n"

                        st.markdown(f"### 🔍 최종 예측 하위 유형: **[{predicted_final}]**")
                        if hybrid_msg:
                            st.warning(hybrid_msg)
                        
                        # Radar Chart
                        labels = list(model_step2.classes_)
                        fig_radar = plt.figure(figsize=(4, 4))
                        ax = fig_radar.add_subplot(111, polar=True)
                        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
                        angles += angles[:1]
                        stats = probs.tolist() + [probs[0]]
                        ax.plot(angles, stats, 'r-', linewidth=2)
                        ax.fill(angles, stats, 'r', alpha=0.25)
                        ax.set_xticks(angles[:-1])
                        ax.set_xticklabels(labels)
                        st.pyplot(fig_radar)
