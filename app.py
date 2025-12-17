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
# [중요] 변수 전역 설정
# ==========================================
FEATS_STEP1 = ['F0', 'Range', 'Intensity', 'SPS', 'VHI_Total', 'VHI_P', 'VHI_F', 'VHI_E']
FEATS_STEP2 = FEATS_STEP1 + ['P_Pitch', 'P_Range', 'P_Loudness', 'P_Rate', 'P_Artic']

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
    
    if os.path.exists(DATA_FILE) or os.path.exists("training_data.xlsx"):
        loaders = [
            (lambda f: pd.read_excel(f.replace(".csv", ".xlsx")), "excel"),
            (lambda f: pd.read_csv(f, encoding='utf-8'), "utf-8"),
            (lambda f: pd.read_csv(f, encoding='cp949'), "cp949"),
            (lambda f: pd.read_csv(f, encoding='euc-kr'), "euc-kr")
        ]
        
        target_file = "training_data.xlsx" if os.path.exists("training_data.xlsx") else DATA_FILE
        
        df_raw = None
        for loader, enc_name in loaders:
            try:
                df_raw = loader(target_file)
                if df_raw is not None and not df_raw.empty: break
            except: continue
                
        if df_raw is not None:
            try:
                data_list = []
                for _, row in df_raw.iterrows():
                    label = str(row.get('진단결과 (Label)', 'Normal')).strip()
                    
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
                        continue

                    # VHI 점수 체계 보정
                    raw_total = row.get('VHI총점', 0)
                    raw_p = row.get('VHI_신체', 0)
                    raw_f = row.get('VHI_기능', 0)
                    raw_e = row.get('VHI_정서', 0)
                    
                    if raw_total > 40: 
                        vhi_f = (raw_f / 40.0) * 20.0
                        vhi_p = (raw_p / 40.0) * 12.0
                        vhi_e = (raw_e / 40.0) * 8.0
                        vhi_total = vhi_f + vhi_p + vhi_e
                    else:
                        vhi_total = raw_total
                        vhi_f = raw_f
                        vhi_p = raw_p
                        vhi_e = raw_e
                    
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
                
                df = pd.DataFrame(data_list, columns=FEATS_STEP2 + ['Diagnosis', 'Subgroup'])
                
                for col in FEATS_STEP2[:4]: # Acoustic vars
                    df[col] = df[col].fillna(df[col].mean())
                
                df[FEATS_STEP1[4:]] = df[FEATS_STEP1[4:]].fillna(0) # VHI vars

            except Exception as e:
                st.error(f"데이터 전처리 오류: {e}")
                df = None

    if df is None:
        st.warning("⚠️ 학습 데이터가 없습니다.")
        return None, None

    model_step1 = RandomForestClassifier(n_estimators=200, random_state=42)
    model_step1.fit(df[FEATS_STEP1], df['Diagnosis'])

    df_pd = df[df['Diagnosis'] == 'Parkinson'].copy()
    if not df_pd.empty:
        for col in FEATS_STEP2[8:]:
             df_pd[col] = df_pd[col].fillna(df_pd[col].mean())
             
        model_step2 = RandomForestClassifier(n_estimators=200, random_state=42)
        model_step2.fit(df_pd[FEATS_STEP2], df_pd['Subgroup'])
    else:
        model_step2 = None

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
# [함수] 공통 분석 로직
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

def run_analysis_logic(file_path):
    try:
        fig, f0, rng, dur = plot_pitch_contour_plotly(file_path, 75, 300)
        sound = parselmouth.Sound(file_path)
        intensity = sound.to_intensity()
        mean_db = call(intensity, "Get mean", 0, 0, "energy")
        sps = st.session_state.user_syllables / dur if dur > 0 else 0
        
        smr_events, _ = auto_detect_smr_events(file_path)
        
        st.session_state.update({
            'f0_mean': f0, 'pitch_range': rng, 'mean_db': mean_db, 
            'sps': sps, 'duration': dur, 'fig_plotly': fig, 
            'is_analyzed': True, 'smr_events': smr_events
        })
        return True
    except Exception as e:
        st.error(f"분석 오류: {e}")
        return False

# ==========================================
# [수정된 함수] 종합 해석 생성기
# ==========================================
def generate_interpretation(prob_normal, db, sps, range_val, artic, vhi, vhi_e):
    positives = []
    negatives = []

    # 1. 긍정적 요인 (Normal 확률을 높이는 요소)
    if vhi < 15:
        positives.append(f"환자 본인의 주관적 불편함(VHI {vhi}점)이 낮아 일상 대화에 심리적 부담이 적습니다.")
    if range_val >= 100:
        positives.append(f"음도 범위({range_val:.1f}Hz)가 넓어 목소리에 생동감이 있고 억양이 자연스럽습니다.")
    if artic >= 75:
        positives.append(f"청지각적 조음 정확도({artic}점)가 양호하여 의사소통 명료도가 높습니다.")
    
    # [수정] 말속도가 4.5 미만이면(느리더라도) '긍정적/안정적'으로 평가
    if sps < 4.5:
        positives.append(f"말속도({sps:.2f} SPS)가 급격히 빨라지는 가속 현상 없이 안정적으로 유지되고 있습니다.")
        
    if db >= 60:
        positives.append(f"성량({db:.1f} dB)이 튼튼하여 정상적인 발성이 유지되고 있습니다.")

    # 2. 부정적/위험 요인 (PD 확률을 남기는 요소)
    if db < 60:
        negatives.append(f"성량({db:.1f} dB)이 일반 대화 수준(60dB)보다 작아 파킨슨병의 '강도 감소(Hypophonia)' 특성과 유사합니다.")
    
    # [수정] 말속도가 3.0 미만이어도 문제 삼지 않음 (삭제됨). 4.5 이상일 때만 경고.
    if sps >= 4.5:
        negatives.append(f"말속도({sps:.2f} SPS)가 지나치게 빨라 가속보행(Festination)과 유사한 말속도 가속 징후가 의심됩니다.")
        
    if artic < 70:
        negatives.append(f"발음의 정확도({artic}점)가 다소 낮아 파킨슨병의 조음 문제(Dysarthria) 징후로 해석될 여지가 있습니다.")
    if vhi >= 20:
        negatives.append(f"VHI 점수({vhi}점)가 높아 환자 스스로 음성 문제를 크게 자각하고 있습니다.")
    if vhi_e >= 5:
        negatives.append("정서적 스트레스(VHI-E)가 높아 말하기에 대한 불안감이 감지됩니다.")

    return positives, negatives

# --- UI Title ---
st.title("🧠 파킨슨병(PD) 음성 하위유형 변별 진단 시스템")
st.markdown("청지각(Perceptual) + 음향(Acoustic) + 자가보고(VHI-10) 통합 하이브리드 진단 모델")

# ==========================================
# 1. 문단 낭독 및 음성 분석
# ==========================================
st.header("1. 문단 낭독 및 음성 분석")

if 'user_syllables' not in st.session_state: st.session_state.user_syllables = 80
if 'source_type' not in st.session_state: st.session_state.source_type = None

col_rec, col_up = st.columns(2)

# [좌측: 마이크 녹음]
with col_rec:
    st.markdown("#### 🎙️ 마이크 녹음 & 문단")
    font_size = st.slider("🔍 글자 크기", 15, 50, 28, key="fs_read")
    
    def styled_text(text, size):
        return f"""<div style="font-size: {size}px; line-height: 1.8; border: 1px solid #ddd; padding: 15px; background-color: #f9f9f9; color: #333;">{text}</div>"""

    with st.expander("📖 [1] 산책 문단 (일반용)"):
        full_text = "높은 산에 올라가 맑은 공기를 마시며 소리를 지르면 가슴이 활짝 열리는 듯하다. 바닷가에 나가 조개를 주으며 넓게 펼쳐있는 바다를 바라보면 내 마음 역시 넓어지는 것 같다."
        st.markdown(styled_text(full_text, font_size), unsafe_allow_html=True)
        
    with st.expander("🔎 [2] 바닷가의 추억 (SMR/조음 정밀 진단용)", expanded=True):
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
    
    if st.button("🎙️ 녹음된 음성 분석", key="btn_anal_mic"):
        if audio_buf:
            with open(TEMP_FILENAME, "wb") as f: f.write(audio_buf.read())
            st.session_state.current_wav_path = os.path.join(os.getcwd(), TEMP_FILENAME)
            st.session_state.source_type = "mic"
            run_analysis_logic(st.session_state.current_wav_path)
        else:
            st.warning("먼저 녹음을 진행해주세요.")

# [우측: 파일 업로드]
with col_up:
    st.markdown("#### 📂 파일 업로드")
    up_file = st.file_uploader("WAV 파일 선택", type=["wav"], key="up_read")
    
    if up_file:
        st.audio(up_file, format='audio/wav')
    
    if st.button("📂 업로드 파일 분석", key="btn_anal_file"):
        if up_file:
            with open(TEMP_FILENAME, "wb") as f: f.write(up_file.read())
            st.session_state.current_wav_path = os.path.join(os.getcwd(), TEMP_FILENAME)
            st.session_state.source_type = "upload"
            run_analysis_logic(st.session_state.current_wav_path)
        else:
            st.warning("먼저 파일을 업로드해주세요.")

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

    if st.session_state.get('smr_events'):
        st.markdown("##### 🔎 SMR 자동 분석")
        events = st.session_state['smr_events']
        smr_df_data = []
        words = ["바닷가", "파도가", "무지개", "바둑이", "보트가", "버터구이", "포토카드", "부탁해", "돋보기", "빈대떡"]
        for i, ev in enumerate(events):
            label = words[i] if i < len(words) else f"구간 {i+1}"
            status = "🟢 양호" if ev['depth'] >= 20 else ("🟡 주의" if ev['depth'] >= 15 else "🔴 불량")
            smr_df_data.append({"단어": label, "폐쇄 깊이(dB)": f"{ev['depth']:.1f}", "상태": status})
        st.dataframe(pd.DataFrame(smr_df_data).T)

    # ==========================================
    # 3. 청지각/자가보고 (VHI-10)
    # ==========================================
    st.markdown("---")
    st.subheader("3. 청지각 평가 및 자가보고 (VHI-10)")
    
    cc1, cc2 = st.columns([1, 1.2])
    
    with cc1:
        st.markdown("#### 🔊 청지각 평가")
        p_artic = st.slider("조음 정확도 (Articulation)", 0, 100, 50, help="78점 이상이면 정상으로 간주됩니다.")
        p_pitch = st.slider("음도 (Pitch)", 0, 100, 50)
        p_prange = st.slider("음도 범위 (Pitch Range)", 0, 100, 50)
        p_loud = st.slider("강도 (Loudness)", 0, 100, 50)
        p_rate = st.slider("말속도 (Rate)", 0, 100, 50)
        
    with cc2:
        st.markdown("#### 📝 VHI-10 (자가보고)")
        st.caption("0: 전혀, 1: 거의X, 2: 가끔, 3: 자주, 4: 항상")
        
        vhi_opts = [0, 1, 2, 3, 4]
        
        with st.expander("VHI-10 문항 입력 (클릭)", expanded=True):
            # 기능(F)
            q1 = st.select_slider("1. 상대방이 내 말을 알아듣기 힘들어한다", options=vhi_opts)
            q2 = st.select_slider("2. 시끄러운 곳에서 이해하기 어려워한다", options=vhi_opts)
            q5 = st.select_slider("5. 음성문제로 생활에 제한을 받는다", options=vhi_opts)
            q7 = st.select_slider("7. 대화에 끼지 못해 소외감을 느낀다", options=vhi_opts)
            q8 = st.select_slider("8. 음성 문제로 수입 감소가 생긴다", options=vhi_opts)
            
            # 신체(P)
            q3 = st.select_slider("3. 사람들이 목소리가 왜 그러냐고 묻는다", options=vhi_opts)
            q4 = st.select_slider("4. 목소리를 내려면 힘을 주어야 한다", options=vhi_opts)
            q6 = st.select_slider("6. 목소리가 언제 맑게 나올지 알 수 없다", options=vhi_opts)

            # 정서(E)
            q9 = st.select_slider("9. 내 목소리 문제로 속이 상한다", options=vhi_opts)
            q10 = st.select_slider("10. 음성 문제가 장애로 여겨진다", options=vhi_opts)

        # 영역별 계산
        vhi_f = q1 + q2 + q5 + q7 + q8
        vhi_p = q3 + q4 + q6
        vhi_e = q9 + q10
        vhi_total = vhi_f + vhi_p + vhi_e
        
        st.divider()
        c_v1, c_v2, c_v3, c_v4 = st.columns(4)
        c_v1.metric("VHI 총점", f"{vhi_total}점", "/ 40")
        c_v2.metric("기능(F)", f"{vhi_f}점", "/ 20")
        c_v3.metric("신체(P)", f"{vhi_p}점", "/ 12")
        c_v4.metric("정서(E)", f"{vhi_e}점", "/ 8")

    # ==========================================
    # 4. 최종 진단 (Hybrid Logic)
    # ==========================================
    st.markdown("---")
    st.subheader("4. 최종 종합 진단")
    
    if st.button("🚀 진단 결과 확인", key="btn_diag"):
        if model_step1 is None:
            st.error("모델 로드 실패. 데이터를 확인하세요.")
        else:
            # Step 0: Rule-based (규칙 기반)
            if p_artic >= 78 and vhi_total < 12:
                st.success(f"🟢 **정상 음성 (Normal) (100.0%)**")
                prob_normal = 100.0
                
                final_decision = "Normal"
                final_db = st.session_state['mean_db'] + db_adj
                final_sps = st.session_state.user_syllables / sel_dur
            
            else:
                # Step 1: 1차 AI 진단
                input_step1 = pd.DataFrame([[
                    st.session_state['f0_mean'], range_adj, final_db, final_sps,
                    vhi_total, vhi_p, vhi_f, vhi_e
                ]], columns=FEATS_STEP1)
                
                pred_1 = model_step1.predict(input_step1)[0]
                prob_1 = model_step1.predict_proba(input_step1)[0]
                
                classes_1 = list(model_step1.classes_)
                normal_idx = classes_1.index('Normal') if 'Normal' in classes_1 else 0
                prob_normal = prob_1[normal_idx] * 100

                if pred_1 == 'Normal':
                    st.success(f"🟢 **정상 음성 (Normal) ({prob_normal:.1f}%)**")
                    final_decision = "Normal"
                
                else:
                    # Step 2: 2차 AI 진단
                    st.error(f"🔴 **파킨슨병(PD) 음성 특성**이 감지되었습니다.")
                    st.write("1차 AI 진단 결과 파킨슨 패턴과 유사합니다. 세부 유형을 분석합니다.")
                    
                    if model_step2:
                        input_step2 = pd.DataFrame([[
                            st.session_state['f0_mean'], range_adj, final_db, final_sps,
                            vhi_total, vhi_p, vhi_f, vhi_e,
                            p_pitch, p_prange, p_loud, p_rate, p_artic
                        ]], columns=FEATS_STEP2)
                        
                        pred_subtype = model_step2.predict(input_step2)[0]
                        probs_sub = model_step2.predict_proba(input_step2)[0]
                        
                        # --- [Hybrid Logic] 임계값 및 가중치 적용 ---
                        final_decision = pred_subtype
                        warn_msg = []
                        
                        is_rate_feature = False
                        
                        emotional_ratio = vhi_e / 8.0 
                        if emotional_ratio >= 0.55: 
                            is_rate_feature = True
                            warn_msg.append("⚠️ **[중요]** 높은 정서적 스트레스(VHI-정서)가 감지되었습니다. 이는 **'말속도 집단'**의 특징입니다.")
                        
                        # [수정] 객관적 말속도가 빠를 때만 경고
                        if final_sps >= 4.5:
                             is_rate_feature = True
                             warn_msg.append("⚠️ 객관적 말속도(SPS)가 빠릅니다.")
                        
                        if is_rate_feature and "말속도" not in final_decision:
                            final_decision = "말속도 집단 (재조정됨)"
                            warn_msg.append("💡 객관적 지표에 따라 진단 결과가 **'말속도 집단'**으로 보정되었습니다.")

                        # 강도 집단 판별
                        MIC_INTENSITY_CUTOFF = 60.0
                        if final_db < MIC_INTENSITY_CUTOFF:
                            if "강도" not in final_decision:
                                warn_msg.append(f"⚠️ **[중요]** 음성 강도가 {final_db:.1f}dB로 기준보다 낮습니다. **'강도 집단'** 특성이 강합니다.")
                                final_decision = "강도 집단 (재조정됨)"

                        # 조음 집단 판별
                        if vhi_total < 15 and p_artic < 60:
                            if "조음" not in final_decision:
                                warn_msg.append("⚠️ 주관적 불편함(VHI)은 적으나 청지각적 조음 문제가 있습니다. **'조음 집단'** 가능성이 높습니다.")
                                final_decision = "조음 집단 (재조정됨)"

                        st.markdown(f"### 🔍 최종 예측 하위 유형: **[{final_decision}]**")
                        for msg in warn_msg: st.warning(msg)
                        
                        labels = list(model_step2.classes_)
                        labels_with_probs = [f"{label}\n({prob*100:.1f}%)" for label, prob in zip(labels, probs_sub)]
                        
                        fig_radar = plt.figure(figsize=(4, 4))
                        ax = fig_radar.add_subplot(111, polar=True)
                        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
                        angles += angles[:1]
                        stats = probs_sub.tolist() + [probs_sub[0]]

                        ax.plot(angles, stats, linewidth=2, linestyle='solid', color='red')
                        ax.fill(angles, stats, 'red', alpha=0.25)
                        ax.set_xticks(angles[:-1])
                        ax.set_xticklabels(labels_with_probs)
                        
                        c_chart, c_desc = st.columns([1, 2])
                        with c_chart: st.pyplot(fig_radar)
                        with c_desc:
                            if "강도" in final_decision:
                                st.info("💡 **특징:** 목소리 크기가 작고 약합니다. (Hypophonia)")
                            elif "말속도" in final_decision:
                                st.info("💡 **특징:** 말이 빠르거나 리듬이 불규칙하며, 정서적 불안감이 동반될 수 있습니다.")
                            else:
                                st.info("💡 **특징:** 발음이 뭉개지고 정확도가 떨어집니다.")

            # 💡 상세 종합 해석
            st.divider()
            with st.expander("💡 상세 종합 해석 (AI Interpretation) 보기", expanded=True):
                positives, negatives = generate_interpretation(prob_normal, final_db, final_sps, range_adj, p_artic, vhi_total, vhi_e)
                
                st.markdown(f"**1. 정상(Normal) 확률이 {prob_normal:.1f}%로 나타난 이유 (긍정적 요인):**")
                if positives:
                    for p in positives:
                        st.markdown(f"- ✅ {p}")
                else:
                    st.markdown("- 특별한 긍정적 요인이 감지되지 않았습니다.")

                st.markdown(f"**2. 파킨슨(PD) 가능성이 {100-prob_normal:.1f}% 존재하는 이유 (위험 요인):**")
                if negatives:
                    for n in negatives:
                        st.markdown(f"- ⚠️ {n}")
                else:
                    st.markdown("- 특별한 위험 요인이 감지되지 않았습니다.")
                
                # 종합 결론
                if prob_normal >= 70:
                    st.info("📋 **결론:** 전반적으로 양호한 상태이나, 위에서 언급된 일부 '위험 요인'(특히 강도나 조음)에 대해서는 지속적인 관찰이나 가벼운 훈련이 권장됩니다.")
                elif prob_normal >= 40:
                    st.warning("📋 **결론:** 정상과 파킨슨 특성이 혼재되어 있습니다. 특히 경고가 뜬 항목(강도, 속도 등)에 대해 정밀 검사가 필요합니다.")
                else:
                    st.error("📋 **결론:** 파킨슨병의 음성학적 특징이 뚜렷하게 관찰됩니다. 전문의와의 상담 및 음성 치료가 적극 권장됩니다.")
