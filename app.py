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
from datetime import datetime
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
# 0. 머신러닝 모델 학습
# ==========================================
@st.cache_resource
def train_models():
    DATA_FILE = "training_data.csv"
    df = None
    
    if os.path.exists(DATA_FILE):
        loaders = [
            (lambda f: pd.read_csv(f, encoding='utf-8'), "utf-8"),
            (lambda f: pd.read_csv(f, encoding='cp949'), "cp949"),
            (lambda f: pd.read_csv(f, encoding='euc-kr'), "euc-kr"),
            (lambda f: pd.read_excel(f), "excel")
        ]
        
        df_raw = None
        for loader, enc_name in loaders:
            try:
                df_raw = loader(DATA_FILE)
                break
            except:
                continue
                
        if df_raw is not None:
            try:
                data_list = []
                for _, row in df_raw.iterrows():
                    label = str(row['진단결과 (Label)']).strip()
                    
                    if label.lower() == 'normal':
                        diagnosis = "Normal"
                        subgroup = "None"
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
                        continue

                    vhi_total = row['VHI총점']
                    vhi_p = row['VHI_신체']
                    vhi_f = row['VHI_기능']
                    vhi_e = row['VHI_정서']
                    
                    if vhi_total > 40: 
                        vhi_p = vhi_p / 3
                        vhi_f = vhi_f / 3
                        vhi_e = vhi_e / 3
                    
                    p_pitch = row.get('음도(청지각)', 50)
                    p_prange = row.get('음도범위(청지각)', 50)
                    p_loud = row.get('강도(청지각)', 0)
                    p_rate = row.get('말속도(청지각)', 0)
                    p_artic = row.get('조음정확도(청지각)', 0)
                    
                    if pd.isna(p_pitch): p_pitch = 50
                    if pd.isna(p_prange): p_prange = 50
                    if pd.isna(p_loud): p_loud = 0
                    if pd.isna(p_rate): p_rate = 0
                    if pd.isna(p_artic): p_artic = 0

                    data_list.append([
                        row['F0'], row['Range'], row['강도(dB)'], row['SPS'],
                        vhi_p, vhi_f, vhi_e,
                        p_pitch, p_prange, p_loud, p_rate, p_artic,
                        diagnosis, subgroup
                    ])
                
                df = pd.DataFrame(data_list, columns=[
                    'F0', 'Range', 'Intensity', 'SPS', 'VHI_P', 'VHI_F', 'VHI_E', 
                    'P_Pitch', 'P_Range', 'P_Loudness', 'P_Rate', 'P_Artic', 
                    'Diagnosis', 'Subgroup'
                ])
                
            except Exception as e:
                st.error(f"데이터 전처리 오류: {e}")
                df = None
        else:
            st.error("❌ 데이터 파일을 읽을 수 없습니다.")

    if df is None:
        N_SAMPLES = 50
        normal_data = []
        for _ in range(N_SAMPLES):
            normal_data.append([
                151.32, 91.68, 70.0, 4.25,
                0, 0, 0, 50, 50, 85, 50, 95, "Normal", "None"
            ])
        pd_data = []
        for _ in range(N_SAMPLES):
             pd_data.append([
                153.21, 101.21, 50.0, 4.05,
                7, 6, 6, 40, 40, 30, 50, 60, "Parkinson", "강도 집단"
            ])
        df = pd.DataFrame(normal_data + pd_data, columns=[
            'F0', 'Range', 'Intensity', 'SPS', 'VHI_P', 'VHI_F', 'VHI_E', 
            'P_Pitch', 'P_Range', 'P_Loudness', 'P_Rate', 'P_Artic', 
            'Diagnosis', 'Subgroup'
        ])
        st.warning("⚠️ 학습 데이터 로드 실패. 임시 모델 사용.")

    features = ['F0', 'Range', 'Intensity', 'SPS', 'VHI_P', 'VHI_F', 'VHI_E', 
                'P_Pitch', 'P_Range', 'P_Loudness', 'P_Rate', 'P_Artic']

    model_diagnosis = RandomForestClassifier(n_estimators=200, random_state=42)
    model_diagnosis.fit(df[features], df['Diagnosis'])

    df_pd = df[df['Diagnosis'] == 'Parkinson']
    model_subgroup = RandomForestClassifier(n_estimators=200, random_state=42)
    model_subgroup.fit(df_pd[features], df_pd['Subgroup'])

    return model_diagnosis, model_subgroup

try:
    diagnosis_model, subgroup_model = train_models()
except:
    diagnosis_model, subgroup_model = None, None

# --- Sidebar ---
with st.sidebar:
    st.title("👤 대상자 정보")
    subject_name = st.text_input("이름", "대상자")
    subject_age = st.number_input("나이", 1, 120, 60)
    subject_gender = st.selectbox("성별", ["남", "여", "기타"])

    def generate_filename(name, age, gender, task="read", is_uploaded=False):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        type_str = "업로드" if is_uploaded else "녹음"
        gender_short = gender[0] if gender else "X"
        return f"{timestamp}_{name}_{age}세_{gender_short}_{task}_{type_str}.wav"

TEMP_FILENAME = "temp_for_analysis.wav"

# ==========================================
# [함수] 자동 조음 분석 (SMR 1~10 탐지)
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
            
            burst = 0
            if p_idx + 10 < len(values):
                slope = np.max(np.gradient(values[p_idx:p_idx+10]))
                burst = slope
            
            candidates.append({
                "time": time_point,
                "depth": depth,
                "burst": burst
            })
            
        candidates.sort(key=lambda x: x['time'])
        results = candidates[:top_n]
        return results, len(candidates)

    except Exception as e:
        return [], 0

# ==========================================
# [함수] 피치 컨투어 시각화
# ==========================================
def plot_pitch_contour_plotly(sound_path, f0_min, f0_max):
    try:
        sound = parselmouth.Sound(sound_path)
        pitch = call(sound, "To Pitch", 0.0, f0_min, f0_max)
        pitch_array = np.array(pitch.selected_array)
        pitch_values = np.array(pitch_array['frequency'], dtype=np.float64)
        duration = sound.get_total_duration()
        n_points = len(pitch_values)
        time_array = np.linspace(0, duration, n_points)
        
        valid_indices = pitch_values != 0
        valid_times = time_array[valid_indices]
        valid_pitch = pitch_values[valid_indices]

        if len(valid_pitch) > 0:
            median_f0 = np.median(valid_pitch)
            clean_mask = (valid_pitch <= median_f0 + 3 * np.std(valid_pitch)) & (valid_pitch >= median_f0 - 3 * np.std(valid_pitch))
            final_times = valid_times[clean_mask]
            final_pitch = valid_pitch[clean_mask]
            cleaned_mean_f0 = np.mean(final_pitch)
        else:
            final_times = valid_times
            final_pitch = valid_pitch
            cleaned_mean_f0 = 0

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=final_times, y=final_pitch,
            mode='markers', name='Pitch (Hz)',
            marker=dict(size=4, color='red'),
            hovertemplate='시간: %{x:.2f}초<br>음도: %{y:.1f}Hz'
        ))
        fig.update_layout(
            title=f"음도 컨투어 (Pitch Contour)",
            xaxis_title="시간 (초)", yaxis_title="음도 (Hz)",
            yaxis=dict(range=[0, 300]),
            height=300, margin=dict(l=20, r=20, t=40, b=20),
            showlegend=True
        )
        return fig, cleaned_mean_f0, duration
    except Exception as e:
        return None, 0, 0

# --- 제목 ---
st.title("🧠 파킨슨병(PD) 음성 하위유형 변별 진단 시스템")
st.markdown("""
이 프로그램은 **청지각적 평가**, **음향학적 분석**, **자가보고(VHI-10)** 데이터를 통합하여 
파킨슨병 환자의 음성 특성을 **3가지 하위 유형(강도/말속도/조음 집단)**으로 분류합니다.
""")

# ==========================================
# 1. 문단 낭독 및 음성 분석
# ==========================================
st.header("1. 문단 낭독 및 음성 분석")

if 'user_syllables' not in st.session_state:
    st.session_state.user_syllables = 142

# 낭독 문단 표시
with st.expander("📖 낭독 문단: '바닷가의 추억' (SMR 단어 10개 포함)", expanded=True):
    st.markdown("""
    <div style="font-size: 24px; line-height: 1.8; border: 1px solid #ddd; padding: 20px; background-color: #f9f9f9; color: #333;">
    <strong>바닷가</strong>에 <strong>파도가</strong> 시원하게 밀려옵니다.<br>
    하늘에는 알록달록 <strong>무지개</strong>가 떴고, 귀여운 <strong>바둑이</strong>가 뛰어옵니다.<br>
    저 멀리 하얀 <strong>보트가</strong> 지나가는 것을 보며 <strong>버터구이</strong> 오징어를 먹었습니다.<br>
    친구가 기념으로 <strong>포토카드</strong>를 찍어달라고 <strong>부탁해</strong>서, <br>
    <strong>돋보기</strong>를 쓴 것처럼 자세히 화면을 보고 셔터를 눌렀습니다.<br>
    출출한 배를 달래려 시장에서 <strong>빈대떡</strong>도 사 먹었습니다.
    </div>
    """, unsafe_allow_html=True)
    st.caption("* 굵은 글씨는 SMR(조음교대운동) 분석을 위한 핵심 단어입니다.")

# 녹음/업로드 선택
col_rec, col_up = st.columns(2)
with col_rec:
    audio_buf = st.audio_input("🎙️ 마이크 녹음", label_visibility="visible")
    if audio_buf:
        with open(TEMP_FILENAME, "wb") as f: f.write(audio_buf.read())
        st.session_state.current_wav_path = os.path.join(os.getcwd(), TEMP_FILENAME)
        st.success("녹음 완료")

with col_up:
    up_file = st.file_uploader("📂 WAV 파일 업로드", type=["wav"])
    if up_file:
        with open(TEMP_FILENAME, "wb") as f: f.write(up_file.read())
        st.session_state.current_wav_path = os.path.join(os.getcwd(), TEMP_FILENAME)
        st.success("파일 준비됨")

syllables_rec = st.number_input("전체 음절 수 (기본값: 142)", 1, 500, 142)
st.session_state.user_syllables = syllables_rec

# 분석 버튼
if st.button("🛠️ 음성 분석 실행", key="btn_anal_main"):
    if 'current_wav_path' in st.session_state:
        try:
            # 1. 기본 음향 분석
            fig_plotly, f0_mean, dur = plot_pitch_contour_plotly(st.session_state.current_wav_path, 75, 300)
            
            sound = parselmouth.Sound(st.session_state.current_wav_path)
            pitch = call(sound, "To Pitch", 0.0, 75, 300)
            pitch_vals = pitch.selected_array['frequency']
            valid_p = pitch_vals[pitch_vals != 0]
            pitch_range = np.max(valid_p) - np.min(valid_p) if len(valid_p) > 0 else 0
            
            intensity = sound.to_intensity()
            mean_db = call(intensity, "Get mean", 0, 0, "energy")
            
            sps = st.session_state.user_syllables / dur
            
            # 2. SMR 자동 탐지
            smr_events, smr_count = auto_detect_smr_events(st.session_state.current_wav_path, top_n=10)
            
            # 세션 저장
            st.session_state.update({
                'f0_mean': f0_mean, 'pitch_range': pitch_range,
                'mean_db': mean_db, 'sps': sps, 'duration': dur,
                'fig_plotly': fig_plotly, 'is_analyzed': True,
                'smr_events': smr_events
            })
            
        except Exception as e:
            st.error(f"분석 오류: {e}")
    else:
        st.warning("먼저 녹음을 하거나 파일을 업로드하세요.")

# ==========================================
# 2. 분석 결과 및 보정
# ==========================================
if 'is_analyzed' in st.session_state and st.session_state['is_analyzed']:
    st.markdown("---")
    st.subheader("2. 분석 결과 및 보정")
    
    # 1) 기본 음향 결과 테이블
    c_res1, c_res2 = st.columns([2, 1])
    with c_res1:
        st.plotly_chart(st.session_state['fig_plotly'], use_container_width=True)
    with c_res2:
        st.markdown("##### 📊 음향 지표 & 보정")
        
        db_adj = st.slider("강도(dB) 보정", -50.0, 50.0, -10.0, 1.0)
        final_db = st.session_state['mean_db'] + db_adj
        
        range_adj = st.slider("음도범위(Hz) 보정", 0.0, 300.0, st.session_state['pitch_range'], 0.1)
        
        st.markdown("---")
        st.caption("⏱️ **말속도(SPS) 발화 구간 선택**")
        s_time, e_time = st.slider("구간 조절", 0.0, st.session_state['duration'], (0.0, st.session_state['duration']), 0.01, label_visibility="collapsed")
        sel_dur = max(0.1, e_time - s_time)
        final_sps = st.session_state.user_syllables / sel_dur
        
        res_df = pd.DataFrame({
            "항목": ["강도 (dB)", "음도 (F0)", "음도 범위", "말속도 (SPS)"],
            "값": [
                f"{final_db:.2f}", 
                f"{st.session_state['f0_mean']:.2f} Hz", 
                f"{range_adj:.2f} Hz", 
                f"{final_sps:.2f}"
            ]
        })
        st.dataframe(res_df, hide_index=True)

    # 2) SMR 단어 자동 분석 결과
    st.markdown("---")
    st.markdown("### 🔎 SMR 핵심 단어 자동 분석 (1번 ~ 10번)")
    st.info("AI가 녹음된 파일에서 **조음(폐쇄/파열)이 발생하는 주요 구간 10곳**을 자동으로 추출했습니다.")
    
    if 'smr_events' in st.session_state and st.session_state['smr_events']:
        events = st.session_state['smr_events']
        
        smr_data = []
        for i, ev in enumerate(events):
            word_guess = ["바닷가", "파도가", "무지개", "바둑이", "보트가", "버터구이", "포토카드", "부탁해", "돋보기", "빈대떡"]
            label = word_guess[i] if i < len(word_guess) else f"구간 {i+1}"
            
            status = "🟢 양호"
            if ev['depth'] < 15: status = "🔴 불량 (소리 샘)"
            elif ev['depth'] < 20: status = "🟡 주의"
            
            smr_data.append({
                "순서": i+1,
                "추정 단어": label,
                "시간 (초)": f"{ev['time']:.2f}",
                "폐쇄 명확도 (dB)": f"{ev['depth']:.1f}",
                "파열 강도": f"{ev['burst']:.1f}",
                "상태": status
            })
            
        st.dataframe(pd.DataFrame(smr_data))
        
        avg_depth = np.mean([e['depth'] for e in events])
        st.metric("평균 폐쇄 명확도", f"{avg_depth:.1f} dB", "20dB 이상 권장")
    else:
        st.warning("분석 가능한 SMR 구간을 찾지 못했습니다.")

    # ==========================================
    # 3. 청지각/자가보고 및 AI 진단
    # ==========================================
    st.markdown("---")
    st.subheader("3. 청지각 평가 및 자가보고 (VHI-10)")
    
    c_input1, c_input2 = st.columns(2)
    
    with c_input1:
        st.markdown("#### 🔊 청지각 평가 (Clinician)")
        p_pitch = st.slider("1. 음도 (Pitch)", 0, 100, 50, help="0(낮음) ~ 100(높음)")
        p_prange = st.slider("2. 음도 범위 (Pitch Range)", 0, 100, 50, help="0(단조로움) ~ 100(변화큼)")
        p_loud = st.slider("3. 강도 (Loudness)", 0, 100, 50, help="0(작음) ~ 100(큼)")
        p_rate = st.slider("4. 말속도 (Rate)", 0, 100, 50, help="0(느림) ~ 100(빠름)")
        p_artic = st.slider("5. 조음 정확도 (Articulation)", 0, 100, 50, help="0(부정확) ~ 100(명확)")
        
    with c_input2:
        st.markdown("#### 📝 VHI-10 자가보고 (Patient)")
        vhi_labels = {0: "0: 전혀", 1: "1: 거의X", 2: "2: 가끔", 3: "3: 자주", 4: "4: 항상"}
        
        st.caption("🔵 **기능 (Functional)**")
        q1 = st.select_slider("F1. 목소리 때문에 상대방이 내 말을 알아듣기 힘들어한다", options=[0,1,2,3,4], format_func=lambda x: vhi_labels[x])
        q2 = st.select_slider("F3. 시끄러운 곳에서는 사람들이 내 말을 이해하기 어려워한다", options=[0,1,2,3,4], format_func=lambda x: vhi_labels[x])
        q5 = st.select_slider("F16. 음성문제로 개인 생활과 사회생활에 제한을 받는다", options=[0,1,2,3,4], format_func=lambda x: vhi_labels[x])
        q7 = st.select_slider("F19. 내 목소리 때문에 대화에 끼지 못하여 소외감을 느낀다", options=[0,1,2,3,4], format_func=lambda x: vhi_labels[x])
        q8 = st.select_slider("F22. 음성 문제로 인해 소득(수입)에 감소가 생긴다", options=[0,1,2,3,4], format_func=lambda x: vhi_labels[x])
        vhi_f = q1 + q2 + q5 + q7 + q8

        st.caption("🔴 **신체 (Physical)**")
        q3 = st.select_slider("P10. 사람들이 나에게 목소리가 왜 그러냐고 묻는다", options=[0,1,2,3,4], format_func=lambda x: vhi_labels[x])
        q4 = st.select_slider("P14. 목소리를 내려면 힘을 주어야 나오는 것 같다", options=[0,1,2,3,4], format_func=lambda x: vhi_labels[x])
        q6 = st.select_slider("P17. 목소리가 언제쯤 맑게 잘 나올지 알 수가 없다", options=[0,1,2,3,4], format_func=lambda x: vhi_labels[x])
        vhi_p = q3 + q4 + q6

        st.caption("🟡 **정서 (Emotional)**")
        q9 = st.select_slider("E23. 내 목소리 문제로 속이 상한다", options=[0,1,2,3,4], format_func=lambda x: vhi_labels[x])
        q10 = st.select_slider("E25. 음성 문제가 장애로(핸디캡으로) 여겨진다", options=[0,1,2,3,4], format_func=lambda x: vhi_labels[x])
        vhi_e = q9 + q10

        vhi_total = vhi_f + vhi_p + vhi_e
        st.info(f"**VHI 총점: {vhi_total}/40점** (기능 {vhi_f}, 신체 {vhi_p}, 정서 {vhi_e})")

    # ==========================================
    # 4. 최종 진단
    # ==========================================
    st.markdown("---")
    st.subheader("4. 최종 종합 진단")
    
    if st.button("🚀 진단 결과 확인", key="btn_diag"):
        if diagnosis_model:
            input_vec = pd.DataFrame([[
                st.session_state['f0_mean'], range_adj, final_db, final_sps,
                vhi_p, vhi_f, vhi_e, p_pitch, p_prange, p_loud, p_rate, p_artic
            ]], columns=['F0', 'Range', 'Intensity', 'SPS', 'VHI_P', 'VHI_F', 'VHI_E', 
                         'P_Pitch', 'P_Range', 'P_Loudness', 'P_Rate', 'P_Artic'])
            
            diag = diagnosis_model.predict(input_vec)[0]
            probs = diagnosis_model.predict_proba(input_vec)[0]
            
            if diag == 'Normal':
                st.success(f"🟢 **정상 음성 (Normal)** 범위입니다. (확률: {probs[0]*100:.1f}%)")
            else:
                st.error(f"🔴 **파킨슨병(PD) 음성** 특성이 감지되었습니다. (확률: {probs[1]*100:.1f}%)")
                
                sub_pred = subgroup_model.predict(input_vec)[0]
                sub_probs = subgroup_model.predict_proba(input_vec)[0]
                classes = subgroup_model.classes_
                
                fig_radar = plt.figure(figsize=(5, 5))
                ax = fig_radar.add_subplot(111, polar=True)
                
                stats = sub_probs.tolist() + [sub_probs[0]]
                angles = np.linspace(0, 2*np.pi, len(classes), endpoint=False).tolist() + [0]
                
                ax.plot(angles, stats, linewidth=2, linestyle='solid', color='red')
                ax.fill(angles, stats, 'red', alpha=0.25)
                
                labels_with_pct = [f"{cls}\n({prob*100:.1f}%)" for cls, prob in zip(classes, sub_probs)]
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(labels_with_pct, size=11, fontweight='bold')
                ax.set_yticklabels([])
                ax.set_title("하위 유형 확률 분포", size=15, pad=20)
                
                c_fig, c_txt = st.columns([1, 1])
                with c_fig:
                    st.pyplot(fig_radar)
                with c_txt:
                    st.write(f"### 가장 유력한 유형: **[{sub_pred}]**")
                    if sub_pred == "강도 집단":
                        st.info("💡 **임상적 제언:** 목소리 크기가 현저히 작고 힘이 없습니다. (Hypophonia)")
                    elif sub_pred == "말속도 집단":
                        st.info("💡 **임상적 제언:** 말이 빨라지거나 리듬이 불규칙합니다. (Festination)")
                    else:
                        st.info("💡 **임상적 제언:** 발음이 뭉개지고 정확도가 떨어집니다. (Dysarthria)")
