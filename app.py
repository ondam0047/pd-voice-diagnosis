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

                    # VHI 스케일링
                    vhi_total = row['VHI총점']
                    vhi_p = row['VHI_신체']
                    vhi_f = row['VHI_기능']
                    vhi_e = row['VHI_정서']
                    
                    if vhi_total > 40: 
                        vhi_p = vhi_p / 3
                        vhi_f = vhi_f / 3
                        vhi_e = vhi_e / 3
                    
                    p_loud = row['강도(청지각)'] if pd.notnull(row['강도(청지각)']) else 0
                    p_rate = row['말속도(청지각)'] if pd.notnull(row['말속도(청지각)']) else 0
                    p_artic = row['조음정확도(청지각)'] if pd.notnull(row['조음정확도(청지각)']) else 0

                    data_list.append([
                        row['F0'], row['Range'], row['강도(dB)'], row['SPS'],
                        vhi_p, vhi_f, vhi_e,
                        p_loud, p_rate, p_artic,
                        diagnosis, subgroup
                    ])
                
                df = pd.DataFrame(data_list, columns=[
                    'F0', 'Range', 'Intensity', 'SPS', 'VHI_P', 'VHI_F', 'VHI_E', 
                    'P_Loudness', 'P_Rate', 'P_Artic', 'Diagnosis', 'Subgroup'
                ])
                
            except Exception as e:
                st.error(f"데이터 전처리 오류: {e}")
                df = None
        else:
            st.error("❌ 데이터 파일을 읽을 수 없습니다.")

    if df is None:
        # 비상용 가상 데이터
        N_SAMPLES = 50
        normal_data = []
        for _ in range(N_SAMPLES):
            normal_data.append([
                np.random.normal(151.32, 25.0), np.random.normal(91.68, 20.0), np.random.normal(70.0, 5.0), np.random.normal(4.25, 0.8),
                0, 0, 0, np.random.normal(85.0, 10.0), np.random.normal(50.0, 10.0), np.random.normal(95.0, 5.0), "Normal", "None"
            ])
        pd_data = []
        for _ in range(N_SAMPLES):
             pd_data.append([
                np.random.normal(153.21, 25.0), np.random.normal(101.21, 25.0), np.random.normal(50.0, 5.0), np.random.normal(4.05, 0.8),
                7, 6, 6, 30, 50, 60, "Parkinson", "강도 집단"
            ])
        df = pd.DataFrame(normal_data + pd_data, columns=[
            'F0', 'Range', 'Intensity', 'SPS', 'VHI_P', 'VHI_F', 'VHI_E', 
            'P_Loudness', 'P_Rate', 'P_Artic', 'Diagnosis', 'Subgroup'
        ])
        st.warning("⚠️ 학습 데이터 로드 실패. 임시 모델 사용.")

    features = ['F0', 'Range', 'Intensity', 'SPS', 'VHI_P', 'VHI_F', 'VHI_E', 'P_Loudness', 'P_Rate', 'P_Artic']

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

    def generate_filename(name, age, gender, is_uploaded=False):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        type_str = "업로드" if is_uploaded else "녹음"
        gender_short = gender[0] if gender else "X"
        return f"{timestamp}_{name}_{age}세_{gender_short}_{type_str}.wav"

TEMP_FILENAME = "temp_for_analysis.wav"

# ==========================================
# 피치 컨투어 시각화 함수
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
        return fig, cleaned_mean_f0, duration
    except Exception as e:
        st.error(f"피치 컨투어 오류: {e}")
        return None, 0, 0

# --- 제목 ---
st.title("🧠 파킨슨병(PD) 음성 하위유형 변별 진단 시스템")
st.markdown("""
이 프로그램은 **청지각적 평가**, **음향학적 분석**, **자가보고(VHI-10)** 데이터를 통합하여 
파킨슨병 환자의 음성 특성을 **3가지 하위 유형(강도/말속도/조음 집단)**으로 분류합니다.
**현재 모델은 업로드된 실제 임상 데이터를 기반으로 학습되었습니다.**
""")

# ==========================================
# 1. 음성 녹음 및 업로드
# ==========================================
st.header("1. 음성 녹음 및 파일 업로드")

tab1, tab2 = st.tabs(["🎙️ 마이크 녹음 (시작/중지)", "📂 파일 업로드"])

if 'current_wav_path' in st.session_state:
    current_wav_path = st.session_state.current_wav_path

if 'user_syllables' not in st.session_state:
    st.session_state.user_syllables = 75 

# [Tab 1] 마이크 녹음
with tab1:
    st.markdown("##### 📜 낭독 문단 선택")
    
    # 글자 크기 조절
    font_size = st.slider("🔍 글자 크기 조절", min_value=15, max_value=50, value=28)
    
    def styled_text(text, size):
        return f"""
        <div style="
            font-size: {size}px; 
            line-height: 1.8; 
            border: 1px solid #ddd; 
            padding: 20px; 
            border-radius: 10px; 
            background-color: #f9f9f9;
            color: #333;">
            {text}
        </div>
        """

    # [문단 1] 산책 문단
    with st.expander("📖 [1] 산책 문단 (일반용) - 클릭해서 열기"):
        st.caption("✅ 권장 총 음절 수: **69개** (아래 입력창에 69를 입력하세요)")
        san_chaek_text = """
        높은 산에 올라가 맑은 공기를 마시며 소리를 지르면 가슴이 활짝 열리는 듯하다.<br><br>
        바닷가에 나가 조개를 주으며 넓게 펼쳐있는 바다를 바라보면 내 마음 역시 넓어지는 것 같다.
        """
        st.markdown(styled_text(san_chaek_text, font_size), unsafe_allow_html=True)

    # [문단 2] 사계절의 소리 (수정됨: 줄글 형태)
    with st.expander("🔎 [2] 사계절의 소리 (정밀 진단용) - 클릭해서 열기"):
        st.caption("✅ 권장 총 음절 수: **75개** (아래 입력창에 75를 입력하세요)")
        four_seasons_text = """
        따뜻한 봄바람이 불면 빨간 튤립이 톡톡 터집니다.<br>
        파란 파도가 바닷가 바위를 덮칩니다.<br>
        높은 하늘 아래 단풍잎이 뚝뚝 떨어집니다.<br>
        추운 겨울밤, 팥죽 한 그릇을 뚝딱 비웠습니다.
        """
        st.markdown(styled_text(four_seasons_text, font_size), unsafe_allow_html=True)

    st.markdown("---")
    
    syllables_rec = st.number_input("낭독한 문단의 총 음절 수 (위 권장 수치 참고)", min_value=1, value=75, key="syllables_rec")
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
    
    syllables_up = st.number_input("낭독 문단의 총 음절 수 (업로드 파일용)", min_value=1, value=75, key="syllables_up")
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
            
            # 1. Pitch Plotly
            fig_plotly, f0_mean_calc, total_duration = plot_pitch_contour_plotly(current_wav_path, 75, 300)
            
            # 2. Pitch Range
            pitch = call(sound, "To Pitch", 0.0, 75, 300)
            pitch_vals = pitch.selected_array['frequency']
            valid_p = pitch_vals[pitch_vals != 0]
            if len(valid_p) > 0:
                pitch_range_init = np.max(valid_p) - np.min(valid_p)
            else:
                pitch_range_init = 0

            # 3. Intensity
            intensity = sound.to_intensity()
            mean_db_spl = call(intensity, "Get mean", 0, 0, "energy")
            
            # SPS
            sps = st.session_state.user_syllables / total_duration
            
            st.session_state['pitch_range_init'] = pitch_range_init
            st.session_state['f0_mean_init'] = f0_mean_calc
            st.session_state['mean_db_spl_init'] = mean_db_spl
            st.session_state['sps_init'] = sps
            st.session_state['fig_plotly'] = fig_plotly
            st.session_state['total_duration'] = total_duration
            st.session_state['is_analyzed'] = True
            
            st.success(f"✅ 분석 완료 (적용된 음절 수: {st.session_state.user_syllables})")
        except Exception as e:
            st.error(f"분석 오류: {e}")

if 'is_analyzed' in st.session_state and st.session_state['is_analyzed']:
    st.markdown("#### 🎧 음도 컨투어 및 발화 구간 선택")
    
    if 'fig_plotly' in st.session_state:
        st.plotly_chart(st.session_state['fig_plotly'])
    
    # 발화 구간 수동 설정
    st.markdown("##### ⏱️ 말속도(SPS) 계산을 위한 발화 구간 설정")
    total_dur = st.session_state['total_duration']
    start_time, end_time = st.slider(
        "발화 구간 선택 (초)",
        min_value=0.0, max_value=float(total_dur),
        value=(0.0, float(total_dur)), step=0.01
    )
    selected_duration = end_time - start_time
    if selected_duration < 0.1: selected_duration = 0.1
    recalc_sps = st.session_state.user_syllables / selected_duration
    st.session_state['sps_init'] = recalc_sps 
    st.info(f"선택된 시간: **{start_time:.2f}초 ~ {end_time:.2f}초** (총 **{selected_duration:.2f}초**)  👉  재계산된 말속도: **{recalc_sps:.2f} SPS**")

    st.markdown("---")
    st.markdown("##### 🎚️ 기기적 측정값 보정 (Calibration)")
    
    c1, c2 = st.columns(2)
    with c1:
        db_offset = st.slider("🔊 강도(dB) 보정", -50.0, 50.0, -10.0, 1.0)
        final_db = st.session_state['mean_db_spl_init'] + db_offset
    with c2:
        slider_min, slider_max = 0.0, 300.0
        default_val = st.session_state['pitch_range_init']
        if default_val > slider_max: default_val = slider_max
        if default_val < slider_min: default_val = slider_min
        final_pitch_range = st.slider("🎵 음도 범위(Range) 보정", slider_min, slider_max, default_val, 0.1)
    
    st.markdown("#### 📊 최종 음향 분석 지표")
    acoustic_data = {
        "지표명": ["강도 (dB)", "음도 (F0)", "음도 범위", "말속도 (SPS)"],
        "값": [
            f"{final_db:.2f} dB (보정됨)",
            f"{st.session_state['f0_mean_init']:.2f} Hz",
            f"{final_pitch_range:.2f} Hz",
            f"{recalc_sps:.2f}" 
        ]
    }
    df_acoustic = pd.DataFrame(acoustic_data)
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
        if diagnosis_model is None:
            st.error("🚨 학습 데이터 파일(training_data.csv)이 GitHub에 없어서 모델을 만들지 못했습니다.")
        else:
            feature_names = ['F0', 'Range', 'Intensity', 'SPS', 'VHI_P', 'VHI_F', 'VHI_E', 'P_Loudness', 'P_Rate', 'P_Artic']
            
            input_values = [[
                st.session_state['f0_mean_init'],
                final_pitch_range,
                final_db, 
                recalc_sps, 
                vhi_physical,
                vhi_functional,
                vhi_emotional,
                p_loudness,     
                p_rate,         
                p_articulation  
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
                st.subheader("🔍 2단계: 하위 유형 분류 (3대 유형)")
                st.write(f"가장 유력한 유형은 **[{sub_pred}]** 입니다.")
                
                fig = plt.figure(figsize=(4, 4)) 
                ax = fig.add_subplot(111, polar=True)
                
                if platform.system() != 'Windows':
                    plt.rc('font', family='NanumGothic')

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
                    desc = "말속도가 빠르거나 불규칙하며, 정서적 스트레스가 높게 나타납니다."
                else: 
                    desc = "청지각적 조음 정확도가 현저히 낮고 발음이 불명료한 것이 주된 특징입니다."
                    
                st.info(f"💡 **임상적 제언:** {desc}")
