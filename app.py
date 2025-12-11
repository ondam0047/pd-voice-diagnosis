import streamlit as st
import parselmouth
from parselmouth.praat import call
import numpy as np
import pandas as pd
import plotly.graph_objects as go  # Interactive plotting
import matplotlib.pyplot as plt    
import matplotlib.font_manager as fm 
import os
import platform
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

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
# 0. 머신러닝 모델 학습 (청지각적 VAS 통계 완벽 반영)
# ==========================================
@st.cache_resource
def train_models():
    SCALE_FACTOR = 3.0 
    
    # Feature 순서: [F0, Range, Intensity, SPS, VHI_P, VHI_F, VHI_E, P_Loudness, P_Rate, P_Artic]
    
    # 모델의 안정적인 학습을 위해 각 집단별로 충분한 수의 가상 데이터를 생성합니다 (각 50개)
    # 통계치는 제공해주신 실제 연구 데이터를 따릅니다.
    
    # A. 정상 그룹
    normal_data = []
    for _ in range(50):
        normal_data.append([
            np.random.normal(151.32, 20.0), # F0
            np.random.normal(91.68, 20.0),  # Range
            np.random.normal(70.0, 4.0),    # Intensity
            np.random.normal(4.25, 0.8),    # SPS
            0, 0, 0,                        # VHI
            np.random.normal(80.0, 10.0),   # P_Loudness (정상 범위)
            np.random.normal(50.0, 10.0),   # P_Rate (보통)
            np.random.normal(95.0, 5.0),    # P_Artic (명료함)
            "Normal", "None"
        ])
        
    # B. 파킨슨 그룹 (제공된 통계치 적용)
    pd_data = []
    
    # 1) 강도 집단 (Red)
    # 특징: P_Loudness(강도)가 29.47로 매우 낮음
    for _ in range(50):
        pd_data.append([
            np.random.normal(153.21, 25.0), 
            np.random.normal(101.21, 25.0), 
            np.random.normal(52.0, 5.0),     # 음향 강도도 낮게 설정
            np.random.normal(4.05, 0.8),     
            np.random.normal(20.18 / SCALE_FACTOR, 2.0), 
            np.random.normal(19.36 / SCALE_FACTOR, 2.0), 
            np.random.normal(18.91 / SCALE_FACTOR, 2.0),
            np.random.normal(29.47, 10.0),   # [핵심] P_Loudness: 매우 낮음 (29.47)
            np.random.normal(49.73, 8.89),   # P_Rate: 보통
            np.random.normal(49.53, 15.0),   # P_Artic: 보통 낮음
            "Parkinson", "강도 집단"
        ])
        
    # 2) 말속도 집단 (Yellow)
    # 특징: P_Rate(말속도)가 75.63으로 매우 높음(빠름)
    for _ in range(50):
        pd_data.append([
            np.random.normal(162.90, 25.0), 
            np.random.normal(84.84, 15.0), 
            np.random.normal(60.0, 4.0),     
            np.random.normal(6.0, 0.5),      # 음향 SPS도 빠르게
            np.random.normal(24.67 / SCALE_FACTOR, 2.0), 
            np.random.normal(29.00 / SCALE_FACTOR, 2.0), 
            np.random.normal(32.00 / SCALE_FACTOR, 2.0), 
            np.random.normal(51.56, 13.23),  # P_Loudness: 보통
            np.random.normal(75.63, 10.0),   # [핵심] P_Rate: 매우 빠름 (75.63)
            np.random.normal(56.22, 17.64),  # P_Artic: 보통
            "Parkinson", "말속도 집단"
        ])
        
    # 3) 조음 집단 (Blue)
    # 특징: P_Artic(조음)이 40.97로 가장 낮음. P_Loudness는 65.61로 양호.
    for _ in range(50):
        pd_data.append([
            np.random.normal(151.32, 20.0),  
            np.random.normal(91.68, 20.0),   
            np.random.normal(65.0, 4.0),     
            np.random.normal(4.18, 0.6),     
            np.random.normal(17.75 / SCALE_FACTOR, 2.0), 
            np.random.normal(13.75 / SCALE_FACTOR, 2.0), 
            np.random.normal(11.25 / SCALE_FACTOR, 2.0), 
            np.random.normal(65.61, 5.0),    # P_Loudness: 높음 (65.61) - 강도 집단과 확실히 구별됨
            np.random.normal(50.61, 9.78),   # P_Rate: 보통
            np.random.normal(40.97, 8.0),    # [핵심] P_Artic: 가장 낮음 (40.97)
            "Parkinson", "조음 집단"
        ])

    df = pd.DataFrame(normal_data + pd_data, columns=[
        'F0', 'Range', 'Intensity', 'SPS', 'VHI_P', 'VHI_F', 'VHI_E', 
        'P_Loudness', 'P_Rate', 'P_Artic', 'Diagnosis', 'Subgroup'
    ])

    features = ['F0', 'Range', 'Intensity', 'SPS', 'VHI_P', 'VHI_F', 'VHI_E', 'P_Loudness', 'P_Rate', 'P_Artic']

    model_diagnosis = RandomForestClassifier(n_estimators=100, random_state=42)
    model_diagnosis.fit(df[features], df['Diagnosis'])

    df_pd = df[df['Diagnosis'] == 'Parkinson']
    model_subgroup = RandomForestClassifier(n_estimators=100, random_state=42)
    model_subgroup.fit(df_pd[features], df_pd['Subgroup'])

    return model_diagnosis, model_subgroup

diagnosis_model, subgroup_model = train_models()

# --- Sidebar ---
with st.sidebar:
    st.title("👤 대상자 정보")
    subject_name = st.text_input("이름", "대상자")
    subject_age = st.number_input("나이", 1, 120, 60)
    subject_gender = st.selectbox("성별", ["남", "여", "기타"])

    def generate_filename(name, age, gender, is_uploaded=False):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        type_str = "업로드" if is_uploaded else "녹음"
        gender_short = gender[0] if gender else "X"
        return f"{timestamp}_{name}_{age}세_{gender_short}_{type_str}.wav"

TEMP_FILENAME = "temp_for_analysis.wav"

# ==========================================
# 피치 컨투어 시각화 함수 (Plotly)
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
            std_f0 = np.std(valid_pitch)
            upper = median_f0 + 3 * std_f0
            lower = median_f0 - 3 * std_f0
            clean_mask = (valid_pitch <= upper) & (valid_pitch >= lower)
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
        fig.add_trace(go.Scatter(
            x=[0, duration], y=[cleaned_mean_f0, cleaned_mean_f0],
            mode='lines', name=f'평균 F0 ({cleaned_mean_f0:.1f}Hz)',
            line=dict(color='gray', dash='dash'), hoverinfo='skip'
        ))
        fig.update_layout(
            title=f"음도 컨투어 (Pitch Contour)",
            xaxis_title="시간 (초)", yaxis_title="음도 (Hz)",
            yaxis=dict(range=[0, 300]),
            height=300, margin=dict(l=20, r=20, t=40, b=20),
            showlegend=True
        )
        return fig, cleaned_mean_f0
    except Exception as e:
        st.error(f"피치 컨투어 오류: {e}")
        return None, 0

# --- 제목 ---
st.title("🧠 파킨슨병(PD) 음성 하위유형 변별 진단 시스템")
st.markdown("""
이 프로그램은 **청지각적 평가**, **음향학적 분석**, **자가보고(VHI-10)** 데이터를 통합하여 
파킨슨병 환자의 음성 특성을 4가지 하위 유형으로 분류합니다.
""")

# ==========================================
# 1. 음성 녹음 및 업로드
# ==========================================
st.header("1. 음성 녹음 및 파일 업로드")

tab1, tab2 = st.tabs(["🎙️ 마이크 녹음 (시작/중지)", "📂 파일 업로드"])

if 'current_wav_path' in st.session_state:
    current_wav_path = st.session_state.current_wav_path

if 'user_syllables' not in st.session_state:
    st.session_state.user_syllables = 69 

# [Tab 1] 마이크 녹음
with tab1:
    st.markdown("##### 마이크 녹음 (시작/중지)")
    st.caption("아래 마이크 아이콘을 눌러 녹음을 시작하고, 완료되면 정지 버튼을 누르세요.")
    
    syllables_rec = st.number_input("낭독 문단의 총 음절 수", min_value=1, value=69, key="syllables_rec")
    st.session_state.user_syllables = syllables_rec

    audio_buffer = st.audio_input("녹음하기", label_visibility="collapsed")
    
    if audio_buffer:
        audio_bytes = audio_buffer.read()
        with open(TEMP_FILENAME, "wb") as f:
            f.write(audio_bytes)
        
        st.session_state.current_wav_path = os.path.join(os.getcwd(), TEMP_FILENAME)
        final_filename = generate_filename(subject_name, subject_age, subject_gender, is_uploaded=False)
        
        st.success("녹음 완료! 분석 준비됨.")
        st.download_button(
            label="💾 녹음 파일 다운로드",
            data=audio_bytes,
            file_name=final_filename,
            mime="audio/wav"
        )

# [Tab 2] 파일 업로드
with tab2:
    st.markdown("##### 기존 WAV 파일 업로드")
    
    syllables_up = st.number_input("낭독 문단의 총 음절 수 (업로드 파일용)", min_value=1, value=69, key="syllables_up")
    uploaded_file = st.file_uploader("WAV 파일을 선택하세요", type=["wav"], key="file_uploader")
    
    if uploaded_file is not None:
        try:
            file_bytes = uploaded_file.read()
            with open(TEMP_FILENAME, 'wb') as f:
                f.write(file_bytes)
            
            st.session_state.current_wav_path = os.path.join(os.getcwd(), TEMP_FILENAME)
            st.session_state.user_syllables = syllables_up
            
            final_filename = generate_filename(subject_name, subject_age, subject_gender, is_uploaded=True)
            
            st.success("업로드 완료! 분석 준비됨.")
            st.download_button(
                label="💾 업로드 파일 다운로드 (저장)",
                data=file_bytes,
                file_name=final_filename,
                mime="audio/wav"
            )
            
        except Exception as e:
            st.error(f"오류: {e}")

# ==========================================
# 2. 객관적/기기적 평가
# ==========================================
st.header("2. 음향학적 분석 및 수동 보정")

is_analyzed = False

if 'current_wav_path' in st.session_state and st.session_state.current_wav_path and os.path.exists(st.session_state.current_wav_path):
    current_wav_path = st.session_state.current_wav_path
    
    if st.button("🛠️ 음성 분석 실행/갱신", key="analyze_button"):
        try:
            sound = parselmouth.Sound(current_wav_path)
            
            # 1. Pitch Plotly (F0 Mean 포함)
            fig_plotly, f0_mean_calc = plot_pitch_contour_plotly(current_wav_path, 75, 300)
            
            # 2. Pitch Range (Cleaned)
            pitch = call(sound, "To Pitch", 0.0, 75, 300)
            pitch_vals = pitch.selected_array['frequency']
            valid_p = pitch_vals[pitch_vals != 0]
            if len(valid_p) > 0:
                pitch_range_init = np.max(valid_p) - np.min(valid_p)
            else:
                pitch_range_init = 0

            # 3. Intensity, SPS
            intensity = sound.to_intensity()
            mean_db_spl = call(intensity, "Get mean", 0, 0, "energy")
            sps = st.session_state.user_syllables / sound.duration
            
            # 4. Jitter/Shimmer 제거됨
            
            st.session_state['pitch_range_init'] = pitch_range_init
            st.session_state['f0_mean_init'] = f0_mean_calc
            st.session_state['mean_db_spl_init'] = mean_db_spl
            st.session_state['sps_init'] = sps
            st.session_state['fig_plotly'] = fig_plotly
            st.session_state['is_analyzed'] = True
            
            st.success(f"✅ 분석 완료 (적용된 음절 수: {st.session_state.user_syllables})")
        except Exception as e:
            st.error(f"분석 오류: {e}")

if 'is_analyzed' in st.session_state and st.session_state['is_analyzed']:
    st.markdown("#### 🎧 음도 범위 (Pitch Range) 수동 보정")
    
    if 'fig_plotly' in st.session_state:
        st.plotly_chart(st.session_state['fig_plotly'])
    
    col_adj1, col_adj2 = st.columns([2, 1])
    with col_adj1:
        final_pitch_range = st.slider("최종 음도 범위 (Hz) 보정", 0.0, 150.0, st.session_state['pitch_range_init'], 0.1)
    
    st.markdown("#### 📊 최종 음향 분석 지표")
    
    acoustic_data = {
        "지표명": ["강도 (dB)", "음도 (F0)", "음도 범위", "말속도 (SPS)"],
        "값": [
            f"{st.session_state['mean_db_spl_init']:.2f} dB",
            f"{st.session_state['f0_mean_init']:.2f} Hz",
            f"{final_pitch_range:.2f} Hz",
            f"{st.session_state['sps_init']:.2f}"
        ]
    }
    df_acoustic = pd.DataFrame(acoustic_data)
    c_table, c_dummy = st.columns([1, 2])
    with c_table:
        st.dataframe(df_acoustic, hide_index=True)

# ==========================================
# 3. 청지각적 및 자가보고 평가
# ==========================================
st.markdown("---")
st.header("3. 청지각적 및 자가보고 평가")

c1, c2 = st.columns(2)

with c1:
    st.subheader("🔊 청지각적 평가")
    st.caption("대상자의 음성 특성을 평가해주세요 (0 ~ 100)")
    p_pitch = st.slider("음도", 0, 100, 50, help="0(낮다) ~ 100(높다)")
    p_pitch_range = st.slider("음도 범위", 0, 100, 50, help="0(좁다/단조롭다) ~ 100(넓다/변화크다)")
    p_loudness = st.slider("강도", 0, 100, 50, help="0(작다) ~ 100(크다)")
    p_rate = st.slider("말속도", 0, 100, 50, help="0(느리다) ~ 100(빠르다)")
    p_articulation = st.slider("조음 정확도", 0, 100, 50, help="0(나쁘다) ~ 100(좋다)")

with c2:
    st.subheader("📝 환자 자가보고 (VHI-10)")
    vhi_scale = [0, 1, 2, 3, 4]
    vhi_labels = {0: "0: 전혀", 1: "1: 거의X", 2: "2: 가끔", 3: "3: 자주", 4: "4: 항상"}
    def vhi_slider(label, k):
        return st.select_slider(label, options=vhi_scale, value=0, key=k, format_func=lambda x: vhi_labels[x])

    q1 = vhi_slider("1. 전화 통화가 힘들다", 'q1')
    q2 = vhi_slider("2. 대화가 불편하다", 'q2')
    q3 = vhi_slider("3. 목소리가 불안정하다", 'q3')
    q4 = vhi_slider("4. 업무 수행 어려움", 'q4')
    q5 = vhi_slider("5. 목소리가 거칠다", 'q5')
    q6 = vhi_slider("6. 목이 쉽게 피곤하다", 'q6')
    q7 = vhi_slider("7. 목에 힘이 들어간다", 'q7')
    q8 = vhi_slider("8. 자신감이 떨어진다", 'q8')
    q9 = vhi_slider("9. 불안하거나 우울하다", 'q9')
    q10 = vhi_slider("10. 타인의 지적을 받는다", 'q10')

    vhi_functional = q1 + q2 + q4
    vhi_physical = q3 + q5 + q6 + q7
    vhi_emotional = q8 + q9 + q10
    vhi_total = vhi_functional + vhi_physical + vhi_emotional
    st.markdown(f"**VHI 총점: {vhi_total}/40** (신체 {vhi_physical}, 기능 {vhi_functional}, 정서 {vhi_emotional})")

# ==========================================
# 4. 종합 진단 및 분류 결과
# ==========================================
st.markdown("---")
st.header("4. 종합 진단 및 분류 결과")

if st.button("🚀 최종 변별 진단 실행", key="final_classify_button"):
    if 'is_analyzed' not in st.session_state or not st.session_state['is_analyzed']:
        st.error("⚠️ 음성 분석 (2단계)을 먼저 실행해 주세요.")
    else:
        # [수정] 3가지 청지각 변수(강도, 말속도, 조음) 모두 포함
        feature_names = ['F0', 'Range', 'Intensity', 'SPS', 'VHI_P', 'VHI_F', 'VHI_E', 'P_Loudness', 'P_Rate', 'P_Artic']
        
        input_values = [[
            st.session_state['f0_mean_init'],
            final_pitch_range,
            st.session_state['mean_db_spl_init'],
            st.session_state['sps_init'],
            vhi_physical,
            vhi_functional,
            vhi_emotional,
            p_loudness,     # 청지각-강도
            p_rate,         # 청지각-말속도
            p_articulation  # 청지각-조음
        ]]
        
        input_features = pd.DataFrame(input_values, columns=feature_names)
        
        diag_pred = diagnosis_model.predict(input_features)[0]
        diag_prob = diagnosis_model.predict_proba(input_features)[0] 
        
        st.subheader("📊 1단계: 변별 진단 결과")
        
        if diag_pred == "Normal":
            st.success(f"🟢 **정상 음성 (Normal)** 범위에 속합니다.")
            st.metric("정상 확률", f"{diag_prob[0]*100:.1f}%")
            st.info("파킨슨병 특이적 음성 징후가 관찰되지 않았습니다.")
            
        else:
            st.error(f"🔴 **파킨슨병(PD) 음성 장애** 특성이 감지되었습니다.")
            st.metric("PD 의심 확률", f"{diag_prob[1]*100:.1f}%")
            
            sub_pred = subgroup_model.predict(input_features)[0]
            sub_probs = subgroup_model.predict_proba(input_features)[0]
            classes = subgroup_model.classes_
            
            st.markdown("---")
            st.subheader("🔍 2단계: 하위 유형 분류")
            st.write(f"가장 유력한 유형은 **[{sub_pred}]** 입니다.")
            
            fig = plt.figure(figsize=(4, 4)) 
            ax = fig.add_subplot(111, polar=True)
            
            values = sub_probs.tolist()
            values += values[:1] 
            angles = np.linspace(0, 2 * np.pi, len(classes), endpoint=False).tolist()
            angles += angles[:1]
            
            ax.fill(angles, values, color='red', alpha=0.25)
            ax.plot(angles, values, color='red', linewidth=2)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(classes, size=10) 
            ax.set_title("파킨슨 음성 하위 유형 확률", size=12, pad=15)
            
            c_chart, c_empty = st.columns([1, 1]) 
            with c_chart:
                st.pyplot(fig)
            
            if sub_pred == "강도 집단":
                desc = "청지각적 강도가 현저히 낮고(약한 목소리), 신체적 불편함이 주요 특징입니다."
            elif sub_pred == "말속도 집단":
                desc = "말속도가 매우 빠르며(가속보행 현상), 정서적 스트레스가 높게 나타납니다."
            else: # 조음 집단
                desc = "청지각적 조음 정확도가 현저히 낮고 발음이 불명료한 것이 주된 특징입니다."
                
            st.info(f"💡 **임상적 제언:** {desc}")
