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

    def generate_filename(name, age, gender, task="read", is_uploaded=False):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        type_str = "업로드" if is_uploaded else "녹음"
        gender_short = gender[0] if gender else "X"
        return f"{timestamp}_{name}_{age}세_{gender_short}_{task}_{type_str}.wav"

TEMP_FILENAME = "temp_for_analysis.wav"

# ==========================================
# [함수] 자동 조음 분석 (SMR Auto Analysis)
# ==========================================
def auto_analyze_articulation(sound_path):
    """
    1. 문장 분리: 묵음(Pause)을 기준으로 문장을 나눔.
    2. 타겟 선정: '바닷가에 파도가...'가 있는 **첫 번째 문장**을 우선 분석.
    3. 정밀 분석: 해당 구간의 폐쇄 명확도(Stop Gap)와 파열 강도(Burst) 계산.
    """
    try:
        sound = parselmouth.Sound(sound_path)
        intensity = sound.to_intensity(time_step=0.01) # 10ms 단위
        times = intensity.xs()
        values = intensity.values[0, :]
        
        # 1. 문장 분리 (Heuristic: 긴 묵음 > 0.4초 기준)
        threshold = np.max(values) - 25 # 최대 강도 대비 -25dB 이하를 묵음으로 간주
        is_speech = values > threshold
        
        segments = []
        current_segment = []
        for t, sp in zip(times, is_speech):
            if sp:
                current_segment.append(t)
            else:
                if current_segment:
                    if current_segment[-1] - current_segment[0] > 0.5: # 0.5초 이상 유효 발화
                        segments.append((current_segment[0], current_segment[-1]))
                    current_segment = []
        if current_segment:
            if current_segment[-1] - current_segment[0] > 0.5:
                segments.append((current_segment[0], current_segment[-1]))
        
        # 2. 타겟 구간 선정 (1번째 문장에 '바닷가'가 있으므로 첫 번째 구간 선택)
        if len(segments) >= 1:
            target_start, target_end = segments[0]
            target_label = "1번째 문장 ('바닷가에 파도가...')"
        else:
            target_start, target_end = 0, sound.get_total_duration()
            target_label = "전체 구간 (자동 분리 실패)"
            
        # 3. 정밀 분석 (Stop Gap & Burst)
        part = sound.extract_part(from_time=target_start, to_time=target_end)
        part_int = part.to_intensity(time_step=0.002) # 2ms 정밀
        p_vals = part_int.values[0, :]
        
        # 골짜기(Valley) 찾기 -> 폐쇄음 구간
        inv_vals = -p_vals
        valleys, _ = find_peaks(inv_vals, prominence=5, distance=20) 
        
        stop_gap_depths = []
        burst_strengths = []
        
        for v_idx in valleys:
            # Depth Calculation
            v_int = p_vals[v_idx]
            start_search = max(0, v_idx - 50)
            end_search = min(len(p_vals), v_idx + 50)
            local_max = np.max(p_vals[start_search:end_search])
            depth = local_max - v_int
            stop_gap_depths.append(depth)
            
            # Burst Calculation (기울기)
            if v_idx + 10 < len(p_vals):
                slope = np.max(np.gradient(p_vals[v_idx:v_idx+10]))
                burst_strengths.append(slope)
        
        avg_depth = np.mean(stop_gap_depths) if stop_gap_depths else 0
        avg_burst = np.mean(burst_strengths) if burst_strengths else 0
        
        # 스펙트로그램 생성
        spectrogram = part.to_spectrogram()
        X, Y = spectrogram.x_grid(), spectrogram.y_grid()
        sg_db = 10 * np.log10(spectrogram.values)
        
        fig_spec = go.Figure(data=go.Heatmap(
            z=sg_db, x=X, y=Y, colorscale='Viridis', showscale=False
        ))
        fig_spec.update_layout(
            title=f"자동 분석 구간: {target_label}",
            xaxis_title="시간 (초)", yaxis_title="주파수 (Hz)",
            height=250, margin=dict(l=20, r=20, t=30, b=20)
        )
        
        return {
            "avg_depth": avg_depth,
            "avg_burst": avg_burst,
            "fig_spec": fig_spec,
            "label": target_label
        }, None

    except Exception as e:
        return None, str(e)

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
            # Outlier removal
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
            title=f"전체 음도 컨투어 (Pitch Contour)",
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
# 메인 분석 UI
# ==========================================
st.header("1. 문단 낭독 및 음성 분석")

col_rec, col_up = st.columns(2)

# 기본값 설정 (새 문단의 음절 수는 약 142개)
if 'user_syllables' not in st.session_state:
    st.session_state.user_syllables = 142

with col_rec:
    st.markdown("#### 🎙️ 마이크 녹음")
    font_size = st.slider("🔍 글자 크기", 15, 50, 25, key="fs_read")
    
    def styled_text(text, size):
        return f"""<div style="font-size: {size}px; line-height: 1.8; border: 1px solid #ddd; padding: 15px; background-color: #f9f9f9; color: #333;">{text}</div>"""

    with st.expander("📖 [1] 산책 문단 (일반용)"):
        st.caption("권장 음절 수: 69")
        st.markdown(styled_text("높은 산에 올라가 맑은 공기를 마시며 소리를 지르면 가슴이 활짝 열리는 듯하다.<br><br>바닷가에 나가 조개를 주으며 넓게 펼쳐있는 바다를 바라보면 내 마음 역시 넓어지는 것 같다.", font_size), unsafe_allow_html=True)
        
    with st.expander("🔎 [2] 바닷가의 추억 (SMR/조음 정밀 진단용)", expanded=True):
        st.caption("권장 음절 수: 142")
        # 줄글 형태로 깔끔하게 표시
        seaside_text = """
        바닷가에 파도가 시원하게 밀려옵니다.<br>
        하늘에는 알록달록 무지개가 떴고, 귀여운 바둑이가 뛰어옵니다.<br>
        저 멀리 하얀 보트가 지나가는 것을 보며 버터구이 오징어를 먹었습니다.<br>
        친구가 기념으로 포토카드를 찍어달라고 부탁해서, 돋보기를 쓴 것처럼 자세히 화면을 보고 셔터를 눌렀습니다.<br>
        출출한 배를 달래려 시장에서 빈대떡도 사 먹었습니다.
        """
        st.markdown(styled_text(seaside_text, font_size), unsafe_allow_html=True)

    syllables_rec = st.number_input("음절 수 (바닷가=142)", 1, 300, 142, key="syl_rec")
    st.session_state.user_syllables = syllables_rec
    
    audio_buf = st.audio_input("낭독 녹음", label_visibility="collapsed")
    if audio_buf:
        with open(TEMP_FILENAME, "wb") as f: f.write(audio_buf.read())
        st.session_state.current_wav_path = os.path.join(os.getcwd(), TEMP_FILENAME)
        st.success("녹음 완료")

with col_up:
    st.markdown("#### 📂 파일 업로드")
    up_file = st.file_uploader("WAV 파일 선택", type=["wav"], key="up_read")
    if up_file:
        with open(TEMP_FILENAME, "wb") as f: f.write(up_file.read())
        st.session_state.current_wav_path = os.path.join(os.getcwd(), TEMP_FILENAME)
        st.success("파일 준비됨")

# 분석 버튼
if st.button("🛠️ 낭독 분석 실행", key="btn_anal_read"):
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
            
            # 2. 자동 조음 분석 (SMR Auto Analysis)
            smr_res, smr_err = auto_analyze_articulation(st.session_state.current_wav_path)
            
            # 세션 저장
            st.session_state.update({
                'f0_mean': f0_mean, 'pitch_range': pitch_range,
                'mean_db': mean_db, 'sps': sps, 'duration': dur,
                'fig_plotly': fig_plotly, 'is_analyzed': True,
                'smr_res': smr_res
            })
            
        except Exception as e:
            st.error(f"분석 오류: {e}")

# 결과 표시
if 'is_analyzed' in st.session_state and st.session_state['is_analyzed']:
    st.markdown("---")
    st.subheader("2. 분석 결과 및 보정")
    
    st.plotly_chart(st.session_state['fig_plotly'], use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        db_adj = st.slider("강도(dB) 보정", -50.0, 50.0, -10.0, 1.0)
        final_db = st.session_state['mean_db'] + db_adj
    with c2:
        range_adj = st.slider("음도범위(Hz) 보정", 0.0, 300.0, st.session_state['pitch_range'], 0.1)
    
    st.markdown("##### 말속도(SPS) 구간 재설정")
    s_time, e_time = st.slider("전체 발화 구간", 0.0, st.session_state['duration'], (0.0, st.session_state['duration']), 0.01)
    sel_dur = max(0.1, e_time - s_time)
    final_sps = st.session_state.user_syllables / sel_dur
    
    # -----------------------------------------------------
    # [신규] SMR 자동 정밀 분석 섹션
    # -----------------------------------------------------
    st.markdown("---")
    st.markdown("### 🔎 AI 자동 조음 분석 (SMR)")
    
    if st.session_state.get('smr_res'):
        smr = st.session_state['smr_res']
        st.info(f"AI가 **[{smr['label']}]** 구간을 자동으로 감지하여 분석했습니다.")
        st.caption("('바닷가', '파도가' 등 SMR 단어가 포함된 구간의 조음 명확도를 분석합니다)")
        
        c_spec, c_met = st.columns([2, 1])
        with c_spec:
            st.plotly_chart(smr['fig_spec'], use_container_width=True)
        with c_met:
            st.markdown("#### 📊 조음 지표")
            
            # 1. Stop Gap Depth
            gap_val = smr['avg_depth']
            st.metric("폐쇄 명확도 (Depth)", f"{gap_val:.1f} dB", help="자음 발음 시 소리가 얼마나 완벽하게 차단되는지 나타냅니다. (20dB 이상 권장)")
            if gap_val < 15: st.error("⚠️ **폐쇄 불완전:** 소리가 샙니다 (조음장애 의심)")
            elif gap_val < 20: st.warning("⚠️ **주의:** 명확도 다소 낮음")
            else: st.success("🟢 **양호:** 폐쇄음이 명확함")
            
            st.divider()
            
            # 2. Burst Strength
            burst_val = smr['avg_burst']
            st.metric("발음 순발력 (Burst)", f"{burst_val:.1f}", help="자음이 터질 때의 에너지가 얼마나 급격히 상승하는지 나타냅니다.")
            if burst_val < 3: st.caption("⚠️ **주의:** 혀끝 힘/속도 부족")
            
    else:
        st.warning("자동 분석에 실패했습니다. 녹음 상태를 확인하거나 수동으로 다시 시도해주세요.")

    st.markdown("---")
    st.subheader("3. 청지각/자가보고 및 AI 진단")
    
    cc1, cc2 = st.columns(2)
    with cc1:
        st.caption("청지각 평가 (0-100)")
        p_loud = st.slider("강도", 0, 100, 50)
        p_rate = st.slider("말속도", 0, 100, 50)
        p_artic = st.slider("조음 정확도", 0, 100, 50)
    with cc2:
        st.caption("VHI-10 (자가보고)")
        vhi_scores = [st.select_slider(f"문항 {i+1}", [0,1,2,3,4], 0) for i in range(10)]
        vhi_p = sum([vhi_scores[2], vhi_scores[4], vhi_scores[5], vhi_scores[6]])
        vhi_f = sum([vhi_scores[0], vhi_scores[1], vhi_scores[3]])
        vhi_e = sum([vhi_scores[7], vhi_scores[8], vhi_scores[9]])
        st.write(f"VHI 총점: {sum(vhi_scores)} (신체{vhi_p}, 기능{vhi_f}, 정서{vhi_e})")

    if st.button("🚀 AI 종합 진단 실행"):
        if diagnosis_model:
            input_vec = pd.DataFrame([[
                st.session_state['f0_mean'], range_adj, final_db, final_sps,
                vhi_p, vhi_f, vhi_e, p_loud, p_rate, p_artic
            ]], columns=['F0', 'Range', 'Intensity', 'SPS', 'VHI_P', 'VHI_F', 'VHI_E', 'P_Loudness', 'P_Rate', 'P_Artic'])
            
            diag = diagnosis_model.predict(input_vec)[0]
            probs = diagnosis_model.predict_proba(input_vec)[0]
            
            if diag == 'Normal':
                st.success(f"🟢 정상 음성 (확률 {probs[0]*100:.1f}%)")
            else:
                st.error(f"🔴 파킨슨병 의심 (확률 {probs[1]*100:.1f}%)")
                sub_pred = subgroup_model.predict(input_vec)[0]
                sub_probs = subgroup_model.predict_proba(input_vec)[0]
                
                # 레이더 차트
                fig_radar = plt.figure(figsize=(5, 5))
                ax = fig_radar.add_subplot(111, polar=True)
                labels = subgroup_model.classes_
                stats = sub_probs.tolist() + [sub_probs[0]]
                angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist() + [0]
                
                ax.plot(angles, stats, linewidth=2, linestyle='solid', color='red')
                ax.fill(angles, stats, 'red', alpha=0.25)
                
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels([f"{l}\n({p*100:.1f}%)" for l, p in zip(labels, sub_probs)], size=11, fontweight='bold')
                
                ax.set_yticks([0.2, 0.4, 0.6, 0.8])
                ax.set_yticklabels([])
                ax.set_title("하위 유형 확률 분포", size=15, pad=20)
                
                c_fig, c_txt = st.columns(2)
                with c_fig: st.pyplot(fig_radar)
                with c_txt:
                    st.write(f"### 가장 유력한 유형: **{sub_pred}**")
                    if sub_pred == "강도 집단": st.info("특징: 목소리가 작고 약함")
                    elif sub_pred == "말속도 집단": st.info("특징: 말이 빠르거나 가속됨")
                    else: st.info("특징: 발음이 부정확하고 뭉개짐")
