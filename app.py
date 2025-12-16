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
# 0. 머신러닝 모델 학습 (Hybrid Logic 적용)
# ==========================================
@st.cache_resource
def train_models():
    DATA_FILE = "training_data.csv"
    df = None
    
    # 1. 데이터 로드
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

                    # VHI 처리
                    vhi_total = row.get('VHI총점', 0)
                    vhi_p = row.get('VHI_신체', 0)
                    vhi_f = row.get('VHI_기능', 0)
                    vhi_e = row.get('VHI_정서', 0)
                    
                    if vhi_total > 40: # 점수 체계 보정
                        vhi_p = vhi_p / 3
                        vhi_f = vhi_f / 3
                        vhi_e = vhi_e / 3
                        vhi_total = vhi_p + vhi_f + vhi_e
                    
                    # 청지각 변수 처리
                    p_pitch = row.get('음도(청지각)', 0)
                    p_prange = row.get('음도범위(청지각)', 0)
                    p_loud = row.get('강도(청지각)', 0)
                    p_rate = row.get('말속도(청지각)', 0)
                    p_artic = row.get('조음정확도(청지각)', 0)
                    
                    data_list.append([
                        row['F0'], row['Range'], row['강도(dB)'], row['SPS'],
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
                
                # VHI 결측치는 0으로
                vhi_vars = ['VHI_Total', 'VHI_P', 'VHI_F', 'VHI_E']
                df[vhi_vars] = df[vhi_vars].fillna(0)

            except Exception as e:
                st.error(f"데이터 전처리 오류: {e}")
                df = None

    if df is None:
        st.warning("⚠️ 학습 데이터가 없습니다. CSV 파일을 확인해주세요.")
        return None, None

    # --- 모델 학습 시작 ---
    
    # Feature 정의
    # 1단계용: 음향 + VHI (청지각 제외! -> 오진 방지 핵심)
    feats_step1 = ['F0', 'Range', 'Intensity', 'SPS', 'VHI_Total', 'VHI_P', 'VHI_F', 'VHI_E']
    
    # 2단계용: 전체 변수 (청지각 포함 -> 세부 유형 분류용)
    feats_step2 = feats_step1 + ['P_Pitch', 'P_Range', 'P_Loudness', 'P_Rate', 'P_Artic']

    # 1. [Step 1 Model] Normal vs Parkinson (Binary)
    model_step1 = RandomForestClassifier(n_estimators=200, random_state=42)
    model_step1.fit(df[feats_step1], df['Diagnosis'])

    # 2. [Step 2 Model] PD Subtype Classification
    df_pd = df[df['Diagnosis'] == 'Parkinson'].copy()
    
    # PD 데이터 내 청지각 결측치는 평균으로 대치
    perceptual_vars = ['P_Pitch', 'P_Range', 'P_Loudness', 'P_Rate', 'P_Artic']
    for col in perceptual_vars:
        df_pd[col] = df_pd[col].fillna(df_pd[col].mean())
        
    model_step2 = RandomForestClassifier(n_estimators=200, random_state=42)
    model_step2.fit(df_pd[feats_step2], df_pd['Subgroup'])

    return model_step1, model_step2

try:
    model_step1, model_step2 = train_models()
except:
    model_step1, model_step2 = None, None

# --- Sidebar ---
with st.sidebar:
    st.title("👤 대상자 정보")
    subject_name = st.text_input("이름", "대상자")
    subject_age = st.number_input("나이", 1, 120, 60)
    subject_gender = st.selectbox("성별", ["남", "여", "기타"])

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
            burst = 0
            if p_idx + 10 < len(values):
                slope = np.max(np.gradient(values[p_idx:p_idx+10]))
                burst = slope
            candidates.append({"time": time_point, "depth": depth, "burst": burst})
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
        valid_t = times[valid_idx]
        valid_p = pitch_vals[valid_idx]

        if len(valid_p) > 0:
            median = np.median(valid_p)
            std = np.std(valid_p)
            valid_mask = (valid_p <= median + 3*std) & (valid_p >= median - 3*std) & \
                         (valid_p <= f0_max) & (valid_p >= f0_min)
            final_t = valid_t[valid_mask]
            final_p = valid_p[valid_mask]
            
            mean_f0 = np.mean(final_p) if len(final_p) > 0 else 0
            rng = np.max(final_p) - np.min(final_p) if len(final_p) > 0 else 0
        else:
            final_t, final_p = [], []
            mean_f0, rng = 0, 0

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=final_t, y=final_p, mode='markers', marker=dict(size=4, color='red'), name='Pitch'))
        if mean_f0 > 0:
            fig.add_trace(go.Scatter(x=[0, duration], y=[mean_f0, mean_f0], mode='lines', line=dict(color='gray', dash='dash'), name='Mean'))
            
        fig.update_layout(title="음도 컨투어 (Outlier 제거됨)", xaxis_title="Time(s)", yaxis_title="Hz", height=300, yaxis=dict(range=[0, 350]))
        return fig, mean_f0, rng, duration
    except:
        return None, 0, 0, 0

# --- UI Title ---
st.title("🧠 파킨슨병(PD) 음성 하위유형 변별 진단 시스템")
st.markdown("청지각(Perceptual) + 음향(Acoustic) + 자가보고(VHI) 통합 하이브리드 진단 모델")

# ==========================================
# 1. 문단 낭독 및 음성 분석
# ==========================================
st.header("1. 문단 낭독 및 음성 분석")

col_rec, col_up = st.columns(2)

if 'user_syllables' not in st.session_state:
    st.session_state.user_syllables = 80

if 'source_type' not in st.session_state:
    st.session_state.source_type = None

# [좌측: 마이크 녹음 및 문단 보기]
with col_rec:
    st.markdown("#### 🎙️ 마이크 녹음 & 문단")
    font_size = st.slider("🔍 글자 크기", 15, 50, 28, key="fs_read")
    
    def styled_text(text, size):
        return f"""<div style="font-size: {size}px; line-height: 1.8; border: 1px solid #ddd; padding: 15px; background-color: #f9f9f9; color: #333;">{text}</div>"""

    # [문단 1] 산책 문단
    with st.expander("📖 [1] 산책 문단 (일반용) - 클릭해서 열기"):
        st.caption("권장 음절 수: 69")
        st.markdown(styled_text("높은 산에 올라가 맑은 공기를 마시며 소리를 지르면 가슴이 활짝 열리는 듯하다.<br><br>바닷가에 나가 조개를 주으며 넓게 펼쳐있는 바다를 바라보면 내 마음 역시 넓어지는 것 같다.", font_size), unsafe_allow_html=True)
        
    # [문단 2] 바닷가의 추억 (단축형)
    with st.expander("🔎 [2] 바닷가의 추억 (SMR/조음 정밀 진단용) - 클릭해서 열기", expanded=True):
        st.caption("권장 음절 수: 80 (단축됨)")
        seaside_text = """
        <strong>바닷가</strong>에 <strong>파도가</strong> 칩니다.<br>
        <strong>무지개</strong> 아래 <strong>바둑이</strong>가 뜁니다.<br>
        <strong>보트가</strong> 지나가고 <strong>버터구이</strong>를 먹습니다.<br>
        <strong>포토카드</strong>를 <strong>부탁해</strong>서 <strong>돋보기</strong>로 봅니다.<br>
        시장에서 <strong>빈대떡</strong>을 사 먹었습니다.
        """
        st.markdown(styled_text(seaside_text, font_size), unsafe_allow_html=True)

    syllables_rec = st.number_input("전체 음절 수 (기본값: 80)", 1, 500, 80, key="syl_rec")
    st.session_state.user_syllables = syllables_rec
    
    audio_buf = st.audio_input("낭독 녹음")
    if audio_buf:
        with open(TEMP_FILENAME, "wb") as f: f.write(audio_buf.read())
        st.session_state.current_wav_path = os.path.join(os.getcwd(), TEMP_FILENAME)
        st.session_state.source_type = "mic"
        st.success("녹음 완료 (SMR 분석 활성화됨)")

# [우측: 파일 업로드]
with col_up:
    st.markdown("#### 📂 파일 업로드")
    up_file = st.file_uploader("WAV 파일 선택", type=["wav"], key="up_read")
    if up_file:
        with open(TEMP_FILENAME, "wb") as f: f.write(up_file.read())
        st.session_state.current_wav_path = os.path.join(os.getcwd(), TEMP_FILENAME)
        st.session_state.source_type = "upload"
        st.success("파일 준비됨 (SMR 분석 비활성화)")

# 분석 버튼
if st.button("🛠️ 음성 분석 실행", key="btn_anal_main"):
    if 'current_wav_path' in st.session_state:
        try:
            fig, f0, rng, dur = plot_pitch_contour_plotly(st.session_state.current_wav_path, 75, 300)
            sound = parselmouth.Sound(st.session_state.current_wav_path)
            intensity = sound.to_intensity()
            mean_db = call(intensity, "Get mean", 0, 0, "energy")
            sps = st.session_state.user_syllables / dur if dur > 0 else 0
            
            smr_events = []
            if st.session_state.source_type == "mic":
                smr_events, _ = auto_detect_smr_events(st.session_state.current_wav_path)
            
            st.session_state.update({
                'f0_mean': f0, 
                'pitch_range': rng, 
                'mean_db': mean_db, 
                'sps': sps, 
                'duration': dur, 
                'fig_plotly': fig, 
                'is_analyzed': True, 
                'smr_events': smr_events
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
        db_adj = st.slider("강도(dB) 보정", -50.0, 50.0, -10.0)
        final_db = st.session_state['mean_db'] + db_adj
        range_adj = st.slider("음도범위(Hz) 보정", 0.0, 300.0, float(st.session_state['pitch_range']))
        s_time, e_time = st.slider("말속도 측정 구간(초)", 0.0, st.session_state['duration'], (0.0, st.session_state['duration']), 0.01)
        sel_dur = max(0.1, e_time - s_time)
        final_sps = st.session_state.user_syllables / sel_dur
        
        st.dataframe(pd.DataFrame({
            "항목": ["강도(dB)", "음도(Hz)", "음도범위(Hz)", "말속도(SPS)"],
            "값": [f"{final_db:.2f}", f"{st.session_state['f0_mean']:.2f}", f"{range_adj:.2f}", f"{final_sps:.2f}"]
        }), hide_index=True)

    # SMR 결과 표시 (마이크 녹음 시)
    if st.session_state.get('smr_events'):
        st.markdown("##### 🔎 SMR 자동 분석 (주요 조음 구간)")
        events = st.session_state['smr_events']
        smr_df_data = []
        words = ["바닷가", "파도가", "무지개", "바둑이", "보트가", "버터구이", "포토카드", "부탁해", "돋보기", "빈대떡"]
        for i, ev in enumerate(events):
            label = words[i] if i < len(words) else f"구간 {i+1}"
            status = "🟢 양호" if ev['depth'] >= 20 else ("🟡 주의" if ev['depth'] >= 15 else "🔴 불량")
            smr_df_data.append({"단어": label, "폐쇄 깊이(dB)": f"{ev['depth']:.1f}", "상태": status})
        st.dataframe(pd.DataFrame(smr_df_data).T)

    # ==========================================
    # 3. 청지각/자가보고 (VHI) - 요청사항 반영 수정됨
    # ==========================================
    st.markdown("---")
    st.subheader("3. 청지각 평가 및 자가보고 (VHI)")
    
    cc1, cc2 = st.columns([1, 1.2]) # VHI 문항이 길어서 비율 조정
    
    with cc1:
        st.markdown("#### 🔊 청지각 평가")
        # [수정] 중요 표시 별표 제거
        p_artic = st.slider("조음 정확도 (Articulation)", 0, 100, 50, help="78점 이상이면 정상으로 간주됩니다.")
        p_pitch = st.slider("음도 (Pitch)", 0, 100, 50)
        p_prange = st.slider("음도 범위 (Pitch Range)", 0, 100, 50)
        p_loud = st.slider("강도 (Loudness)", 0, 100, 50)
        p_rate = st.slider("말속도 (Rate)", 0, 100, 50)
        
    with cc2:
        st.markdown("#### 📝 VHI-10 (자가보고)")
        st.caption("0: 전혀, 1: 거의X, 2: 가끔, 3: 자주, 4: 항상")
        
        # [수정] 요청하신 10개 문항 적용
        vhi_opts = [0, 1, 2, 3, 4]
        
        q1 = st.select_slider("1. 목소리 때문에 상대방이 내 말을 알아듣기 힘들어한다", options=vhi_opts) # 기능
        q2 = st.select_slider("2. 시끄러운 곳에서는 사람들이 내 말을 이해하기 어려워한다", options=vhi_opts) # 기능
        q3 = st.select_slider("3. 사람들이 나에게 목소리가 왜 그러냐고 묻는다", options=vhi_opts) # 신체
        q4 = st.select_slider("4. 목소리를 내려면 힘을 주어야 나오는 것 같다", options=vhi_opts) # 신체
        q5 = st.select_slider("5. 음성문제로 개인 생활과 사회생활에 제한을 받는다", options=vhi_opts) # 기능
        q6 = st.select_slider("6. 목소리가 언제쯤 맑게 잘 나올지 알 수가 없다(예측이 어렵다)", options=vhi_opts) # 신체
        q7 = st.select_slider("7. 내 목소리 때문에 대화에 끼지 못하여 소외감을 느낀다", options=vhi_opts) # 기능
        q8 = st.select_slider("8. 음성 문제로 인해 소득(수입)에 감소가 생긴다", options=vhi_opts) # 기능
        q9 = st.select_slider("9. 내 목소리 문제로 속이 상한다", options=vhi_opts) # 정서
        q10 = st.select_slider("10. 음성 문제가 장애로(핸디캡으로) 여겨진다", options=vhi_opts) # 정서

        # VHI 영역별 계산 (일반적인 VHI-10 분류 기준 적용)
        # 기능(F): 1, 2, 5, 7, 8
        # 신체(P): 3, 4, 6
        # 정서(E): 9, 10
        vhi_f = q1 + q2 + q5 + q7 + q8
        vhi_p = q3 + q4 + q6
        vhi_e = q9 + q10
        vhi_total = vhi_f + vhi_p + vhi_e
        
        st.info(f"**VHI 총점: {vhi_total} / 40점**")

    # ==========================================
    # 4. 최종 진단 (Hybrid Logic)
    # ==========================================
    st.markdown("---")
    st.subheader("4. 최종 종합 진단")
    
    if st.button("🚀 진단 결과 확인", key="btn_diag"):
        if model_step1 is None:
            st.error("모델 로드 실패. 데이터를 확인하세요.")
        else:
            # ---------------------------------------------------------
            # [Step 0] Rule-based Filtering (안전장치)
            # ---------------------------------------------------------
            if p_artic >= 78:
                # [수정] 풍선 효과 제거됨
                st.success(f"🟢 **정상 음성 (Normal)** 입니다.")
                st.info(f"이유: 조음정확도({p_artic}점)가 정상 기준(78점 이상)을 충족합니다.")
            
            else:
                # ---------------------------------------------------------
                # [Step 1] 1차 AI 진단 (Normal vs PD)
                # ---------------------------------------------------------
                input_step1 = pd.DataFrame([[
                    st.session_state['f0_mean'], range_adj, final_db, final_sps,
                    vhi_total, vhi_p, vhi_f, vhi_e
                ]], columns=['F0', 'Range', 'Intensity', 'SPS', 'VHI_Total', 'VHI_P', 'VHI_F', 'VHI_E'])
                
                pred_1 = model_step1.predict(input_step1)[0]
                prob_1 = model_step1.predict_proba(input_step1)[0]
                
                classes_1 = list(model_step1.classes_)
                normal_idx = classes_1.index('Normal') if 'Normal' in classes_1 else 0
                prob_normal = prob_1[normal_idx] * 100

                if pred_1 == 'Normal':
                    # [수정] 풍선 효과 제거됨
                    st.success(f"🟢 **정상 음성 (Normal)** 범위입니다.")
                    st.info(f"AI 판단: 음향적 특성과 VHI 점수가 정상 범주입니다. (정상 확률: {prob_normal:.1f}%)")
                
                else:
                    # ---------------------------------------------------------
                    # [Step 2] 2차 AI 진단 (PD Subtype)
                    # ---------------------------------------------------------
                    st.error(f"🔴 **파킨슨병(PD) 음성 특성**이 감지되었습니다.")
                    st.write("1차 AI 진단 결과 파킨슨 패턴과 유사합니다. 세부 유형을 분석합니다.")
                    
                    input_step2 = pd.DataFrame([[
                        st.session_state['f0_mean'], range_adj, final_db, final_sps,
                        vhi_total, vhi_p, vhi_f, vhi_e,
                        p_pitch, p_prange, p_loud, p_rate, p_artic
                    ]], columns=['F0', 'Range', 'Intensity', 'SPS', 
                                 'VHI_Total', 'VHI_P', 'VHI_F', 'VHI_E',
                                 'P_Pitch', 'P_Range', 'P_Loudness', 'P_Rate', 'P_Artic'])
                    
                    pred_subtype = model_step2.predict(input_step2)[0]
                    probs_sub = model_step2.predict_proba(input_step2)[0]
                    
                    st.markdown(f"### 🔍 예측 하위 유형: **[{pred_subtype}]**")
                    
                    # Radar Chart
                    labels = list(model_step2.classes_)
                    num_vars = len(labels)
                    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
                    angles += angles[:1]
                    stats = probs_sub.tolist() + [probs_sub[0]]

                    fig_radar = plt.figure(figsize=(4, 4))
                    ax = fig_radar.add_subplot(111, polar=True)
                    ax.plot(angles, stats, linewidth=2, linestyle='solid', color='red')
                    ax.fill(angles, stats, 'red', alpha=0.25)
                    ax.set_xticks(angles[:-1])
                    ax.set_xticklabels(labels, size=10, weight='bold')
                    ax.set_yticklabels([])
                    
                    col_chart, col_desc = st.columns([1, 2])
                    with col_chart:
                        st.pyplot(fig_radar)
                    with col_desc:
                        if "강도" in pred_subtype:
                            st.warning("💡 **특징:** 목소리 크기가 작고 약합니다. (Hypophonia)")
                        elif "말속도" in pred_subtype:
                            st.warning("💡 **특징:** 말이 빠르거나 리듬이 불규칙합니다. (Festination)")
                        else:
                            st.warning("💡 **특징:** 발음이 뭉개지고 정확도가 떨어집니다. (Dysarthria)")

