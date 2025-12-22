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
import datetime
import io

import sqlite3
import hashlib
import json
import uuid
from pathlib import Path

# --- 구글 시트 & 이메일 라이브러리 ---
from google.oauth2 import service_account
import gspread
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="파킨슨병 환자 하위유형 분류 프로그램", layout="wide")

# ==========================================
# [설정] 구글 시트 정보 (Secrets)
# ==========================================
HAS_GCP_SECRETS = True
try:
    SHEET_NAME = st.secrets["gcp_info"]["sheet_name"]
except Exception:
    HAS_GCP_SECRETS = False
    SHEET_NAME = None
    # Secrets가 없어도 분석/DB 저장은 가능하게 유지
    st.warning("⚠️ Secrets(gcp/email) 설정이 없어 구글시트/이메일 전송 기능은 비활성화됩니다. (DB 저장은 가능)")

# ==========================================
# [전역 설정] 폰트 및 변수
# ==========================================
FEATS_STEP1 = ['F0', 'Range', 'Intensity', 'SPS', 'VHI_Total', 'VHI_P', 'VHI_F', 'VHI_E']
FEATS_STEP2 = FEATS_STEP1 + ['P_Pitch', 'P_Range', 'P_Loudness', 'P_Rate', 'P_Artic']

def setup_korean_font():
    system_name = platform.system()
    if system_name == 'Windows':
        font_name = "Malgun Gothic"
    elif system_name == 'Darwin':
        font_name = "AppleGothic"
    else:
        font_name = None
    if font_name:
        plt.rc('font', family=font_name)
    plt.rcParams['axes.unicode_minus'] = False

setup_korean_font()

# ==========================================
# [모델 학습] training_data.csv 기반
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
            except: 
                continue

        if df_raw is not None:
            try:
                data_list = []
                for _, row in df_raw.iterrows():
                    label = str(row.get('진단결과 (Label)', 'Normal')).strip()
                    if 'normal' in label.lower(): diagnosis, subgroup = "Normal", "Normal"
                    elif 'pd_intensity' in label.lower(): diagnosis, subgroup = "Parkinson", "강도 집단"
                    elif 'pd_rate' in label.lower(): diagnosis, subgroup = "Parkinson", "말속도 집단"
                    elif 'pd_articulation' in label.lower(): diagnosis, subgroup = "Parkinson", "조음 집단"
                    else: 
                        continue

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
                        vhi_total, vhi_f, vhi_p, vhi_e = raw_total, raw_f, raw_p, raw_e
                    
                    data_list.append([
                        row.get('F0', 0), row.get('Range', 0), row.get('강도(dB)', 0), row.get('SPS', 0),
                        vhi_total, vhi_p, vhi_f, vhi_e,
                        row.get('음도(청지각)', 0), row.get('음도범위(청지각)', 0), row.get('강도(청지각)', 0),
                        row.get('말속도(청지각)', 0), row.get('조음정확도(청지각)', 0),
                        diagnosis, subgroup
                    ])
                df = pd.DataFrame(data_list, columns=FEATS_STEP2 + ['Diagnosis', 'Subgroup'])
                for col in FEATS_STEP2[:4]: 
                    df[col] = df[col].fillna(df[col].mean())
                df[FEATS_STEP1[4:]] = df[FEATS_STEP1[4:]].fillna(0)
            except Exception: 
                df = None

    if df is None: 
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


# ==========================================
# [DB 저장] SQLite (무료/간편) - 익명 로그 저장
#   ※ Streamlit Cloud/무료 호스팅에서는 파일시스템이 재시작 시 초기화될 수 있습니다.
#     지속 저장이 필요하면 Postgres 같은 외부 DB(무료 티어)를 연결하는 것을 권장합니다.
# ==========================================
DB_PATH = os.environ.get("PD_TOOL_DB_PATH", "pd_tool.db")

@st.cache_resource
def _init_db():
    db_file = Path(DB_PATH)
    conn = sqlite3.connect(db_file.as_posix(), check_same_thread=False, timeout=30)
    # 동시성 완화
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
    except Exception:
        pass

    conn.execute("""
    CREATE TABLE IF NOT EXISTS subjects (
        subject_id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        gender TEXT,
        age INTEGER,
        name TEXT
    );
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS analyses (
        analysis_id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        subject_id TEXT NOT NULL,
        model_version TEXT,
        step1_pd_cutoff REAL,
        step1_p_pd REAL,
        step1_p_normal REAL,
        step1_pred TEXT,
        final_decision TEXT,
        normal_prob REAL,
        f0 REAL,
        pitch_range REAL,
        intensity_db REAL,
        sps REAL,
        vhi_total REAL,
        vhi_p REAL,
        vhi_f REAL,
        vhi_e REAL,
        p_pitch REAL,
        p_prange REAL,
        p_loud REAL,
        p_rate REAL,
        p_artic REAL,
        wav_sha256 TEXT,
        wav_filename TEXT,
        extra_json TEXT,
        FOREIGN KEY(subject_id) REFERENCES subjects(subject_id)
    );
    """)
    conn.commit()
    return conn

def _subject_id_from_info(name: str, age: int, gender: str) -> str:
    """개인정보를 그대로 키로 쓰지 않기 위해 salt+hash로 subject_id 생성"""
    salt = None
    try:
        salt = st.secrets.get("privacy", {}).get("salt", None)
    except Exception:
        salt = None
    if not salt:
        # 배포 환경에서 secrets가 없을 때도 고정된 해시가 나오도록 약한 기본 salt 사용
        salt = "PD_TOOL_DEFAULT_SALT"
    raw = f"{salt}|{str(name).strip()}|{str(age).strip()}|{str(gender).strip()}".encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()[:24]

def _sha256_file(path: str) -> str:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""

def save_to_sqlite(wav_path: str, patient_info: dict, analysis: dict, diagnosis: dict, model_meta=None):
    """분석 결과를 SQLite에 저장 (구글시트/이메일 없이도 동작)"""
    conn = _init_db()
    now = datetime.datetime.now().isoformat(timespec="seconds")

    name = str(patient_info.get("name", "")).strip()
    age = patient_info.get("age", None)
    gender = str(patient_info.get("gender", "")).strip()

    try:
        age_int = int(age) if age is not None and str(age).strip() != "" else None
    except Exception:
        age_int = None

    subject_id = _subject_id_from_info(name, age_int if age_int is not None else "", gender)

    # subjects upsert
    conn.execute(
        """INSERT OR IGNORE INTO subjects(subject_id, created_at, gender, age, name)
               VALUES(?, ?, ?, ?, ?);""",
        (subject_id, now, gender, age_int, name if name else None)
    )

    # wav hash
    wav_sha = _sha256_file(wav_path)
    wav_filename = os.path.basename(wav_path) if wav_path else None

    mv = (model_meta or {}).get("model_version", "unknown")
    step1_cutoff = (model_meta or {}).get("step1_pd_cutoff", None)
    step1_p_pd = (model_meta or {}).get("step1_p_pd", None)
    step1_p_norm = (model_meta or {}).get("step1_p_normal", None)
    step1_pred = (model_meta or {}).get("step1_pred", None)

    extra = (model_meta or {}).get("extra", {})
    extra_json = json.dumps(extra, ensure_ascii=False)

    conn.execute(
        """INSERT INTO analyses(
            created_at, subject_id, model_version,
            step1_pd_cutoff, step1_p_pd, step1_p_normal, step1_pred,
            final_decision, normal_prob,
            f0, pitch_range, intensity_db, sps,
            vhi_total, vhi_p, vhi_f, vhi_e,
            p_pitch, p_prange, p_loud, p_rate, p_artic,
            wav_sha256, wav_filename, extra_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
        (
            now, subject_id, mv,
            step1_cutoff, step1_p_pd, step1_p_norm, step1_pred,
            str(diagnosis.get("final", "")), float(diagnosis.get("normal_prob", 0.0)),
            float(analysis.get("f0", 0.0)), float(analysis.get("range", 0.0)), float(analysis.get("db", 0.0)), float(analysis.get("sps", 0.0)),
            float(analysis.get("vhi_total", 0.0)), float(analysis.get("vhi_p", 0.0)), float(analysis.get("vhi_f", 0.0)), float(analysis.get("vhi_e", 0.0)),
            float(analysis.get("p_pitch", 0.0)), float(analysis.get("p_prange", 0.0)), float(analysis.get("p_loud", 0.0)), float(analysis.get("p_rate", 0.0)), float(analysis.get("p_artic", 0.0)),
            wav_sha, wav_filename, extra_json
        )
    )
    conn.commit()
    return True, f"SQLite 저장 완료 (subject_id={subject_id})"

def db_stats():
    conn = _init_db()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM analyses;")
    n_analyses = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM subjects;")
    n_subjects = cur.fetchone()[0]
    return n_subjects, n_analyses

def fetch_recent_analyses(limit: int = 20):
    conn = _init_db()
    cur = conn.cursor()
    cur.execute(
        """SELECT created_at, subject_id, final_decision, normal_prob, f0, pitch_range, intensity_db, sps
             FROM analyses ORDER BY analysis_id DESC LIMIT ?;""",
        (int(limit),)
    )
    rows = cur.fetchall()
    return pd.DataFrame(rows, columns=["created_at","subject_id","final_decision","normal_prob","f0","pitch_range","intensity_db","sps"])


# ==========================================
# [이메일 전송 함수] 파일명: 이름.wav
# ==========================================
def send_email_and_log_sheet(wav_path, patient_info, analysis, diagnosis):
    if not HAS_GCP_SECRETS:
        return False, "Secrets가 없어 구글시트/이메일 전송이 비활성화되어 있습니다. (SQLite 저장을 사용하세요)"
    try:
        creds = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        )
        gc = gspread.authorize(creds)
        sh = gc.open(SHEET_NAME)
        worksheet = sh.sheet1
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = patient_info['name'].replace(" ", "")
        
        # 구글 시트용 파일명 (상세 정보 포함)
        log_filename = f"{safe_name}_{patient_info['age']}_{patient_info['gender']}_{timestamp}.wav"

        row_data = [
            log_filename,
            patient_info['name'], patient_info['age'], patient_info['gender'],
            analysis['f0'], analysis['range'], analysis['db'], analysis['sps'],
            analysis['vhi_total'], analysis['vhi_p'], analysis['vhi_f'], analysis['vhi_e'],
            analysis['p_artic'], analysis['p_pitch'], analysis['p_loud'], analysis['p_rate'], analysis['p_prange'],
            diagnosis['final'], diagnosis['normal_prob']
        ]
        worksheet.append_row(row_data)

        sender = st.secrets["email"]["sender"]
        password = st.secrets["email"]["password"]
        receiver = st.secrets["email"]["receiver"]

        msg = MIMEMultipart()
        msg['From'] = sender
        msg['To'] = receiver
        
        # 이메일 첨부 파일명: 이름.wav
        email_attach_name = f"{safe_name}.wav"
        msg['Subject'] = f"[PD Data] {email_attach_name}"

        body = f"""
        환자: {patient_info['name']} ({patient_info['age']}/{patient_info['gender']})
        진단: {diagnosis['final']} ({diagnosis['normal_prob']:.1f}%)
        
        * 음성 파일이 첨부되었습니다. ({email_attach_name})
        * 상세 수치는 구글 시트에 저장되었습니다.
        """
        msg.attach(MIMEText(body, 'plain'))

        with open(wav_path, "rb") as f:
            part = MIMEBase("audio", "wav")
            part.set_payload(f.read())
        
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f'attachment; filename="{email_attach_name}"')
        msg.attach(part)

        server = smtplib.SMTP(st.secrets["email"]["smtp_server"], st.secrets["email"]["smtp_port"])
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())
        server.quit()

        return True, "구글 시트 기록 + 이메일 전송 성공"
    except Exception as e:
        return False, str(e)

# ============================
# 이하 원본 분석/UI 로직 (그대로)
# ============================

def plot_pitch_contour_plotly(file_path, f0_min=70, f0_max=500):
    sound = parselmouth.Sound(file_path)
    pitch = sound.to_pitch(time_step=0.01, pitch_floor=f0_min, pitch_ceiling=f0_max)
    f0_values = pitch.selected_array['frequency']
    f0_values = f0_values[f0_values != 0]
    times = pitch.xs()

    if len(f0_values) == 0:
        return None, 0, 0, sound.duration
    
    f0_mean = np.mean(f0_values)
    f0_range = np.max(f0_values) - np.min(f0_values)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=pitch.selected_array['frequency'], mode='lines', name="F0"))
    fig.update_layout(
        title="Pitch Contour",
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        yaxis=dict(range=[0, f0_max]),
        height=250
    )
    return fig, f0_mean, f0_range, sound.duration

def auto_detect_smr_events(file_path):
    try:
        sound = parselmouth.Sound(file_path)
        pitch = sound.to_pitch(time_step=0.01, pitch_floor=60, pitch_ceiling=600)
        values = pitch.selected_array['frequency']
        times = pitch.xs()

        nz = values[values > 0]
        if len(nz) < 5:
            return [], 0
        
        med = np.median(nz)
        low_th = med / 1.9
        hi_th = med * 1.9
        values_f = values.copy()
        values_f[(values_f < low_th) | (values_f > hi_th)] = 0

        changes = np.abs(np.diff(values_f))
        jump_th = np.percentile(changes[changes > 0], 95) if np.any(changes > 0) else 0
        peaks = np.where(changes > jump_th)[0]

        candidates = []
        for p_idx in peaks:
            time_point = times[p_idx]
            v_int = values_f[p_idx]
            start_search = max(0, p_idx - 20)
            end_search = min(len(values_f), p_idx + 20)
            local_max = np.max(values_f[start_search:end_search])
            depth = local_max - v_int
            candidates.append({"time": time_point, "depth": depth})
        candidates.sort(key=lambda x: x['time'])
        return candidates, len(candidates)
    except:
        return [], 0

def run_analysis_logic(file_path):
    try:
        fig, f0, rng, dur = plot_pitch_contour_plotly(file_path, 70, 500)
        sound = parselmouth.Sound(file_path)
        intensity = sound.to_intensity()
        mean_db = call(intensity, "Get mean", 0, 0, "energy")
        sps = st.session_state.user_syllables / dur if dur > 0 else 0
        smr_events, smr_count = auto_detect_smr_events(file_path)
        
        st.session_state.update({
            'f0_mean': f0, 'pitch_range': rng, 'mean_db': mean_db, 
            'sps': sps, 'duration': dur, 'fig_plotly': fig, 
            'smr_events': smr_events, 'smr_count': smr_count,
            'is_analyzed': True, 'is_saved': False
        })
        return True
    except Exception as e:
        st.error(f"분석 오류: {e}"); return False

def generate_interpretation(prob_normal, db, sps, range_val, artic, vhi, vhi_e):
    positives, negatives = [], []
    if vhi < 15: positives.append(f"환자 본인의 주관적 불편함(VHI {vhi}점)이 낮아, 일상 대화에 심리적/기능적 부담이 적은 상태입니다.")
    if range_val >= 100: positives.append(f"음도 범위가 {range_val:.1f}Hz로 넓게 나타나, 목소리에 생동감이 있고 억양의 변화가 자연스럽습니다.")
    if artic >= 75: positives.append(f"청지각적 조음 정확도가 {artic}점으로 양호하여, 상대방이 말을 알아듣기에 명료합니다.")
    if db >= 60: positives.append(f"평균 발화 강도가 {db:.1f}dB로 충분하여, 목소리가 비교적 또렷하게 전달될 수 있습니다.")
    if vhi_e < 5: positives.append(f"정서적 영향(VHI-E {vhi_e}점)이 낮아, 목소리 문제로 인한 스트레스/불안이 상대적으로 적은 편입니다.")

    if vhi >= 20: negatives.append(f"주관적 불편함(VHI {vhi}점)이 높아, 음성 문제로 일상생활에서 불편/부담을 느낄 가능성이 있습니다.")
    if range_val < 70: negatives.append(f"음도 범위가 {range_val:.1f}Hz로 좁아, 억양 변화가 제한되어 단조롭게 들릴 수 있습니다.")
    if artic < 65: negatives.append(f"청지각적 조음 정확도가 {artic}점으로 낮아, 말소리가 뭉개져 들릴 가능성이 있습니다.")
    if db < 55: negatives.append(f"평균 발화 강도가 {db:.1f}dB로 낮아, 목소리가 작거나 약하게 전달될 가능성이 있습니다.")
    if sps > 5.5: negatives.append(f"말속도(SPS)가 {sps:.2f}로 빠른 편이라, 말이 급하게 들리거나 명료도가 떨어질 수 있습니다.")
    if vhi_e >= 7: negatives.append(f"정서적 영향(VHI-E {vhi_e}점)이 높아, 음성 문제로 스트레스/불안이 동반될 수 있습니다.")
    return positives, negatives

# ==========================================
# UI
# ==========================================
st.title("🧠 파킨슨병 환자 하위유형 분류 프로그램")
st.write("음성 파일(.wav)을 업로드하고 간단한 입력 후 분석을 진행하세요.")

if 'user_syllables' not in st.session_state:
    st.session_state.user_syllables = 0
if 'is_analyzed' not in st.session_state:
    st.session_state.is_analyzed = False
if 'is_saved' not in st.session_state:
    st.session_state.is_saved = False

uploaded_file = st.file_uploader("🎤 음성 파일 업로드 (.wav)", type=["wav"])
if uploaded_file:
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state.current_wav_path = file_path

    st.audio(uploaded_file, format="audio/wav")

    st.markdown("### 🧾 기본 정보 입력")
    subject_name = st.text_input("이름", value="")
    subject_age = st.number_input("나이", min_value=1, max_value=120, value=60)
    subject_gender = st.selectbox("성별", ["M", "F"])
    user_syllables = st.number_input("발화한 음절 수(대략)", min_value=1, max_value=500, value=40)
    st.session_state.user_syllables = user_syllables

    st.markdown("---")
    if st.button("📈 음성 분석 실행"):
        ok = run_analysis_logic(file_path)
        if ok:
            st.success("✅ 분석 완료! 아래 결과를 확인하세요.")

    if st.session_state.get('is_analyzed'):
        st.markdown("### 📌 분석 결과")
        st.plotly_chart(st.session_state.fig_plotly, use_container_width=True)

        # 기본 값들
        f0 = st.session_state['f0_mean']
        rng = st.session_state['pitch_range']
        mean_db = st.session_state['mean_db']
        sps = st.session_state['sps']
        dur = st.session_state['duration']
        smr_count = st.session_state.get('smr_count', 0)

        st.write(f"- 평균 F0: {f0:.1f} Hz")
        st.write(f"- 음도 범위: {rng:.1f} Hz")
        st.write(f"- 평균 강도: {mean_db:.1f} dB")
        st.write(f"- SPS(초당 음절 수): {sps:.2f}")
        st.write(f"- 발화 길이: {dur:.2f} s")
        st.write(f"- SMR 이벤트(자동 탐지): {smr_count}회")

        st.markdown("### 🧾 VHI 입력")
        vhi_total = st.number_input("VHI 총점", 0, 120, 0)
        vhi_p = st.number_input("VHI-신체", 0, 40, 0)
        vhi_f = st.number_input("VHI-기능", 0, 40, 0)
        vhi_e = st.number_input("VHI-정서", 0, 40, 0)

        st.markdown("### 👂 청지각 평가 입력")
        p_artic = st.number_input("조음정확도(청지각)", 0, 100, 75)
        p_pitch = st.number_input("음도(청지각)", 0, 100, 50)
        p_prange = st.number_input("음도범위(청지각)", 0, 100, 50)
        p_loud = st.number_input("강도(청지각)", 0, 100, 50)
        p_rate = st.number_input("말속도(청지각)", 0, 100, 50)

        st.markdown("---")
        if st.button("🚀 진단 결과 확인"):
            if model_step1:
                range_adj = rng
                final_db = mean_db
                final_sps = sps

                input_1 = pd.DataFrame([[
                    f0, range_adj, final_db, final_sps,
                    vhi_total, vhi_p, vhi_f, vhi_e
                ]], columns=FEATS_STEP1)

                pred_1 = model_step1.predict(input_1)[0]
                proba_1 = model_step1.predict_proba(input_1)[0]
                classes_1 = list(model_step1.classes_)
                prob_normal = float(proba_1[classes_1.index("Normal")]) * 100 if "Normal" in classes_1 else 0.0

                if pred_1 == "Normal":
                    st.success(f"🟢 정상 음성 (Normal) - {prob_normal:.1f}%")
                    final_decision = "Normal"
                else:
                    st.error(f"🔴 파킨슨 가능성 (PD) - {100 - prob_normal:.1f}%")
                    if model_step2:
                        input_2 = pd.DataFrame([[
                            f0, range_adj, final_db, final_sps,
                            vhi_total, vhi_p, vhi_f, vhi_e,
                            p_pitch, p_prange, p_loud, p_rate, p_artic
                        ]], columns=FEATS_STEP2)

                        pred_2 = model_step2.predict(input_2)[0]
                        final_decision = pred_2
                        st.info(f"➡️ PD 하위 집단 예측: {pred_2}")
                    else:
                        final_decision = "Parkinson"

                pos, neg = generate_interpretation(prob_normal, final_db, final_sps, range_adj, p_artic, vhi_total, vhi_e)
                st.markdown("### 🧾 해석(요약)")
                st.markdown(f"**1. 정상일 확률이 {prob_normal:.1f}%로 나온 이유 (긍정 요인):**")
                if pos:
                    for p in pos:
                        st.markdown(f"- ✅ {p}")
                else:
                    st.markdown("- 특별한 강점 요인이 감지되지 않았습니다.")

                st.markdown(f"**2. 파킨슨(PD) 가능성이 {100-prob_normal:.1f}% 존재하는 이유 (위험 요인):**")
                if neg:
                    for n in neg:
                        st.markdown(f"- ⚠️ {n}")
                else:
                    st.markdown("- 특별한 위험 요인이 감지되지 않았습니다.")

                st.session_state.save_ready_data = {
                    'wav_path': st.session_state.current_wav_path,
                    'patient': {'name': subject_name, 'age': subject_age, 'gender': subject_gender},
                    'analysis': {
                        'f0': f0, 'range': range_adj, 'db': final_db, 'sps': final_sps,
                        'vhi_total': vhi_total, 'vhi_p': vhi_p, 'vhi_f': vhi_f, 'vhi_e': vhi_e,
                        'p_artic': p_artic, 'p_pitch': p_pitch, 'p_loud': p_loud, 'p_rate': p_rate, 'p_prange': p_prange
                    },
                    'diagnosis': {'final': final_decision, 'normal_prob': prob_normal}
                }
            else:
                st.error("모델 로드 실패")

# 전송 버튼
st.markdown("---")
if st.button("☁️ 데이터 전송 (메일+시트)", type="primary"):
    if 'save_ready_data' not in st.session_state:
        st.error("🚨 전송할 데이터가 없습니다. 먼저 [🚀 진단 결과 확인]을 눌러주세요!")
    elif st.session_state.get('is_saved'):
        st.warning("이미 전송된 데이터입니다.")
    else:
        with st.spinner("구글 시트 기록 및 이메일 전송 중..."):
            success, msg = send_email_and_log_sheet(
                st.session_state.save_ready_data['wav_path'], 
                st.session_state.save_ready_data['patient'], 
                st.session_state.save_ready_data['analysis'], 
                st.session_state.save_ready_data['diagnosis']
            )
        if success:
            st.session_state.is_saved = True
            st.success(f"✅ 처리 완료! {msg}")
        else:
            st.error(f"❌ 전송 실패: {msg}")


# ==========================================
# [추가] DB 저장 버튼 (SQLite)
# ==========================================
if st.button("🗄️ DB 저장 (SQLite)", type="secondary"):
    if 'save_ready_data' not in st.session_state:
        st.error("🚨 저장할 데이터가 없습니다. 먼저 [🚀 진단 결과 확인]을 눌러주세요!")
    else:
        with st.spinner("SQLite 저장 중..."):
            ok, msg = save_to_sqlite(
                st.session_state.save_ready_data['wav_path'],
                st.session_state.save_ready_data['patient'],
                st.session_state.save_ready_data['analysis'],
                st.session_state.save_ready_data['diagnosis'],
                model_meta={
                    "model_version": "v1.0",
                    "step1_pd_cutoff": None,
                    "step1_p_pd": None,
                    "step1_p_normal": st.session_state.save_ready_data['diagnosis'].get('normal_prob', 0.0)/100.0,
                    "step1_pred": st.session_state.save_ready_data['diagnosis'].get('final', None),
                    "extra": {"note": "saved_from_streamlit"}
                }
            )
        if ok:
            st.success(f"✅ {msg}")
        else:
            st.error(f"❌ DB 저장 실패: {msg}")

with st.expander("🗄️ DB 현황 / 최근 저장 기록"):
    try:
        n_subj, n_ana = db_stats()
        st.write(f"- Subjects: **{n_subj}**")
        st.write(f"- Analyses: **{n_ana}**")

        df_recent = fetch_recent_analyses(limit=20)
        if not df_recent.empty:
            st.dataframe(df_recent, use_container_width=True)
            csv_bytes = df_recent.to_csv(index=False).encode("utf-8-sig")
            st.download_button("최근 20건 CSV 다운로드", data=csv_bytes, file_name="recent_analyses.csv", mime="text/csv")
        else:
            st.write("최근 기록이 없습니다.")

        # DB 파일 다운로드(서버 파일 접근이 가능한 환경에서만 의미)
        try:
            db_file = Path(DB_PATH)
            if db_file.exists():
                st.download_button(
                    "DB 파일 다운로드 (pd_tool.db)",
                    data=db_file.read_bytes(),
                    file_name=db_file.name,
                    mime="application/octet-stream"
                )
        except Exception:
            pass
    except Exception as e:
        st.write(f"DB 정보를 불러오지 못했습니다: {e}")
