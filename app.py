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

# --- 구글 시트 & 이메일 라이브러리 ---
from google.oauth2 import service_account
import gspread
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders

from sklearn.metrics import confusion_matrix, roc_curve

import sqlite3
import hashlib
import json
from pathlib import Path

from scipy.signal import find_peaks

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# --- 페이지 기본 설정 ---
st.set_page_config(page_title="파킨슨병 환자 하위유형 분류 프로그램", layout="wide")

# ==========================================
# [설정] 구글 시트 정보 (Secrets)
# ==========================================
HAS_GCP_SECRETS = True
try:
    SHEET_NAME = st.secrets["gcp_info"]["sheet_name"]
except:
    st.warning("⚠️ Secrets 설정이 없어 구글시트/이메일 전송은 비활성화됩니다. (SQLite 저장은 사용 가능)")
    SHEET_NAME = None
    HAS_GCP_SECRETS = False

# ==========================================
# [전역 설정] 폰트 및 변수
# ==========================================
FEATS_STEP1 = ['F0', 'Range', 'Intensity', 'SPS', 'VHI_Total', 'VHI_P', 'VHI_F', 'VHI_E', 'Sex']
FEATS_STEP2 = FEATS_STEP1 + ['P_Pitch', 'P_Range', 'P_Loudness', 'P_Rate', 'P_Artic']

def sex_to_num(x):
    """성별을 숫자 feature로 변환: 남/M=1.0, 여/F=0.0, 그 외/결측=0.5"""
    if x is None:
        return 0.5
    s = str(x).strip().lower()
    if s in ["남", "남성", "남자", "m", "male", "man", "1"]:
        return 1.0
    if s in ["여", "여성", "여자", "f", "female", "woman", "0", "2"]:
        return 0.0
    return 0.5


@st.cache_resource

def _youden_cutoff(y_true, scores):
    """Youden's J(민감도+특이도-1)를 최대화하는 threshold 반환"""
    fpr, tpr, thr = roc_curve(y_true, scores)
    j = tpr - fpr
    bi = int(np.argmax(j))
    # sklearn roc_curve의 thr에는 inf가 들어갈 수 있어 방어
    cut = float(thr[bi]) if np.isfinite(thr[bi]) else 0.5
    sens = float(tpr[bi])
    spec = float(1.0 - fpr[bi])
    return cut, sens, spec


@st.cache_data
def compute_cutoffs_from_training(_file_mtime=None):
    """
    training_data.csv/xlsx로부터 Step1/Step2 확률 cut-off를 자동 산출
    - 누수 방지: Leave-One-Out(LOO) OOF 확률로 cut-off 산정
    - Step1: 이항 로지스틱(PD 확률) + Youden cut-off
    - Step2: (PD 내부) 정규화 QDA(reg_param) 확률 + 클래스별(OVR) Youden cut-off
    """
    DATA_FILE = "training_data.csv"
    target_file = "training_data.xlsx" if os.path.exists("training_data.xlsx") else DATA_FILE
    if not os.path.exists(target_file):
        return None

    loaders = [
        (lambda f: pd.read_excel(f), "excel"),
        (lambda f: pd.read_csv(f, encoding='utf-8'), "utf-8"),
        (lambda f: pd.read_csv(f, encoding='cp949'), "cp949"),
        (lambda f: pd.read_csv(f, encoding='euc-kr'), "euc-kr")
    ]
    df_raw = None
    for loader, _ in loaders:
        try:
            df_raw = loader(target_file)
            if df_raw is not None and not df_raw.empty:
                break
        except Exception:
            continue
    if df_raw is None or df_raw.empty:
        return None

    # --- 로우 파싱 ---
    data_list = []
    for _, row in df_raw.iterrows():
        label = str(row.get('진단결과 (Label)', 'Normal')).strip()
        l = label.lower()
        if 'normal' in l:
            diagnosis, subgroup = "Normal", "Normal"
        elif 'pd_intensity' in l:
            diagnosis, subgroup = "Parkinson", "강도 집단"
        elif 'pd_rate' in l:
            diagnosis, subgroup = "Parkinson", "말속도 집단"
        elif 'pd_articulation' in l:
            diagnosis, subgroup = "Parkinson", "조음 집단"
        else:
            continue

        raw_total = pd.to_numeric(row.get('VHI총점', 0), errors="coerce")
        raw_p = pd.to_numeric(row.get('VHI_신체', 0), errors="coerce")
        raw_f = pd.to_numeric(row.get('VHI_기능', 0), errors="coerce")
        raw_e = pd.to_numeric(row.get('VHI_정서', 0), errors="coerce")
        raw_total = float(0 if pd.isna(raw_total) else raw_total)
        raw_p = float(0 if pd.isna(raw_p) else raw_p)
        raw_f = float(0 if pd.isna(raw_f) else raw_f)
        raw_e = float(0 if pd.isna(raw_e) else raw_e)

        # VHI는 UI에서 VHI-10(0~40) 기반으로 입력되므로,
        # training_data의 VHI-30(총점 0~120, 하위척도 0~40)을 VHI-10 스케일로 변환해 사용합니다.
        # UI에서 계산하는 분해(기능 0~20 / 신체 0~12 / 정서 0~8)와 동일하게 맞춥니다.
        if raw_total <= 40 and raw_f <= 20 and raw_p <= 12 and raw_e <= 8:
            vhi_total, vhi_p, vhi_f, vhi_e = raw_total, raw_p, raw_f, raw_e
        else:
            vhi_f = (raw_f / 40.0) * 20.0
            vhi_p = (raw_p / 40.0) * 12.0
            vhi_e = (raw_e / 40.0) * 8.0
            vhi_total = vhi_f + vhi_p + vhi_e

        sex_num = sex_to_num(row.get('성별', None))

        data_list.append([
            row.get('F0', 0), row.get('Range', 0), row.get('강도(dB)', 0), row.get('SPS', 0),
            vhi_total, vhi_p, vhi_f, vhi_e, sex_num,
            row.get('음도(청지각)', 0), row.get('음도범위(청지각)', 0), row.get('강도(청지각)', 0),
            row.get('말속도(청지각)', 0), row.get('조음정확도(청지각)', 0),
            diagnosis, subgroup
        ])

    df = pd.DataFrame(data_list, columns=FEATS_STEP2 + ['Diagnosis', 'Subgroup'])

    # 숫자 변환/결측 처리
    for col in FEATS_STEP2:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # 음향/청지각은 평균으로, VHI는 0으로(입력 누락 대비)
    for col in ['F0', 'Range', 'Intensity', 'SPS',
                'P_Pitch', 'P_Range', 'P_Loudness', 'P_Rate', 'P_Artic', 'Sex']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())
    for col in ['VHI_Total', 'VHI_P', 'VHI_F', 'VHI_E']:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # ---------- Step1: Normal vs PD cut-off (LOO OOF) ----------
    X1 = df[FEATS_STEP1].copy()
    y1 = df["Diagnosis"].astype(str).values

    loo = LeaveOneOut()
    oof_pd = np.zeros(len(df), dtype=float)

    pipe1 = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="lbfgs",
            max_iter=5000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    for tr, te in loo.split(X1, y1):
        pipe1.fit(X1.iloc[tr], df['Diagnosis'].iloc[tr].astype(str).values)
        proba = pipe1.predict_proba(X1.iloc[te])[0]
        cls = pipe1.named_steps['clf'].classes_
        pd_idx = int(np.where(cls == 'Parkinson')[0][0]) if 'Parkinson' in cls else -1
        oof_pd[te[0]] = float(proba[pd_idx]) if pd_idx >= 0 else float(proba[-1])

    step1_cutoff, step1_sens, step1_spec = _youden_cutoff(y1, oof_pd)

    # ---------- Step2: PD 내부 3집단 cut-off (클래스별 OVR, LOO OOF) ----------
    df_pd = df[df["Diagnosis"] == "Parkinson"].copy()
    cutoff_by_class = {}
    step2_report = None

    if len(df_pd) >= 3:
        X2 = df_pd[FEATS_STEP2].copy()
        y2 = df_pd["Subgroup"].astype(str).values
        classes = np.unique(y2)
        class_to_idx = {c: i for i, c in enumerate(classes)}

        oof2 = np.zeros((len(df_pd), len(classes)), dtype=float)

        pipe2 = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", QuadraticDiscriminantAnalysis(reg_param=0.1))
        ])

        for tr, te in loo.split(X2, y2):
            # 혹시라도 특정 fold에서 한 클래스가 사라질 경우를 대비(현 데이터에서는 거의 없음)
            y_tr = y2[tr]
            if len(np.unique(y_tr)) < 2:
                continue
            pipe2.fit(X2.iloc[tr], y_tr)
            proba = pipe2.predict_proba(X2.iloc[te])[0]
            fold_classes = pipe2.named_steps["clf"].classes_
            for j, c in enumerate(fold_classes):
                oof2[te[0], class_to_idx[c]] = float(proba[j])

        # 클래스별 OVR Youden cut-off
        for c in classes:
            y_bin = (y2 == c).astype(int)
            p = oof2[:, class_to_idx[c]]
            if np.all(y_bin == 0) or np.all(y_bin == 1):
                cutoff_by_class[c] = 0.5
                continue
            cut, _, _ = _youden_cutoff(y_bin, p)
            cutoff_by_class[c] = float(cut)

        # 참고용: LOO 기준 혼동행렬(단순 argmax)
        y_pred = [classes[int(np.argmax(oof2[i]))] for i in range(len(df_pd))]
        step2_cm = confusion_matrix(y2, y_pred, labels=list(classes))
        step2_report = {"classes": list(classes), "confusion_matrix": step2_cm.tolist()}

    # Step1 혼동행렬(확률 cut-off 적용)
    y_pred1 = (oof_pd >= step1_cutoff).astype(int)
    step1_cm = confusion_matrix(y1, y_pred1, labels=[0, 1])  # 0=Normal,1=PD

    return {
        "step1_cutoff": float(step1_cutoff),
        "step1_sensitivity": float(step1_sens),
        "step1_specificity": float(step1_spec),
        "step1_confusion_matrix": step1_cm.tolist(),
        "step2_cutoff_by_class": cutoff_by_class,
        "step2_report": step2_report
    }


# ==========================================
# [SQLite 저장] Secrets가 없어도 저장 가능한 로컬 DB
# ==========================================
DB_PATH = os.environ.get("PD_TOOL_DB_PATH", "pd_tool.db")

@st.cache_resource
def _db_conn():
    conn = sqlite3.connect(Path(DB_PATH).as_posix(), check_same_thread=False, timeout=30)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
    except Exception:
        pass
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            analysis_id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            subject_name TEXT,
            subject_age INTEGER,
            subject_gender TEXT,
            wav_filename TEXT,
            wav_sha256 TEXT,
            f0 REAL, pitch_range REAL, intensity_db REAL, sps REAL,
            vhi_total REAL, vhi_p REAL, vhi_f REAL, vhi_e REAL,
            p_pitch REAL, p_prange REAL, p_loud REAL, p_rate REAL, p_artic REAL,
            step1_p_pd REAL, step1_p_normal REAL, step1_cutoff REAL,
            final_decision TEXT, normal_prob REAL
        );
    """)
    conn.commit()
    return conn

def _sha256_file(path: str) -> str:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""

def save_to_sqlite(wav_path: str, patient_info: dict, analysis: dict, diagnosis: dict, step1_meta: dict):
    conn = _db_conn()
    now = datetime.datetime.now().isoformat(timespec="seconds")
    wav_filename = os.path.basename(wav_path) if wav_path else None
    wav_sha = _sha256_file(wav_path) if wav_path else ""
    conn.execute(
        """INSERT INTO analyses(
            created_at, subject_name, subject_age, subject_gender,
            wav_filename, wav_sha256,
            f0, pitch_range, intensity_db, sps,
            vhi_total, vhi_p, vhi_f, vhi_e,
            p_pitch, p_prange, p_loud, p_rate, p_artic,
            step1_p_pd, step1_p_normal, step1_cutoff,
            final_decision, normal_prob
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);""",
        (
            now,
            str(patient_info.get("name","")).strip() or None,
            int(patient_info.get("age")) if str(patient_info.get("age","")).strip() != "" else None,
            str(patient_info.get("gender","")).strip() or None,
            wav_filename, wav_sha,
            float(analysis.get("f0",0.0)), float(analysis.get("range",0.0)), float(analysis.get("db",0.0)), float(analysis.get("sps",0.0)),
            float(analysis.get("vhi_total",0.0)), float(analysis.get("vhi_p",0.0)), float(analysis.get("vhi_f",0.0)), float(analysis.get("vhi_e",0.0)),
            float(analysis.get("p_pitch",0.0)), float(analysis.get("p_prange",0.0)), float(analysis.get("p_loud",0.0)), float(analysis.get("p_rate",0.0)), float(analysis.get("p_artic",0.0)),
            float(step1_meta.get("p_pd",0.0)), float(step1_meta.get("p_normal",0.0)), float(step1_meta.get("cutoff",0.5)),
            str(diagnosis.get("final","")), float(diagnosis.get("normal_prob",0.0))
        )
    )
    conn.commit()

def setup_korean_font():
    system_name = platform.system()
    if system_name == 'Windows':
        try:
            plt.rc('font', family='Malgun Gothic')
        except: pass
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
    """
    training_data.csv/xlsx로부터 모델 학습
    - Step1: 이항 로지스틱(정상 vs PD)
    - Step2: 정규화 QDA(PD 하위집단 3분류)
    """
    DATA_FILE = "training_data.csv"
    target_file = "training_data.xlsx" if os.path.exists("training_data.xlsx") else DATA_FILE
    if not os.path.exists(target_file):
        return None, None

    loaders = [
        (lambda f: pd.read_excel(f), "excel"),
        (lambda f: pd.read_csv(f, encoding='utf-8'), "utf-8"),
        (lambda f: pd.read_csv(f, encoding='cp949'), "cp949"),
        (lambda f: pd.read_csv(f, encoding='euc-kr'), "euc-kr")
    ]
    df_raw = None
    for loader, _ in loaders:
        try:
            df_raw = loader(target_file)
            if df_raw is not None and not df_raw.empty:
                break
        except Exception:
            continue
    if df_raw is None or df_raw.empty:
        return None, None

    data_list = []
    for _, row in df_raw.iterrows():
        label = str(row.get('진단결과 (Label)', 'Normal')).strip()
        l = label.lower()
        if 'normal' in l:
            diagnosis, subgroup = "Normal", "Normal"
        elif 'pd_intensity' in l:
            diagnosis, subgroup = "Parkinson", "강도 집단"
        elif 'pd_rate' in l:
            diagnosis, subgroup = "Parkinson", "말속도 집단"
        elif 'pd_articulation' in l:
            diagnosis, subgroup = "Parkinson", "조음 집단"
        else:
            continue

        raw_total = pd.to_numeric(row.get('VHI총점', 0), errors="coerce")
        raw_p = pd.to_numeric(row.get('VHI_신체', 0), errors="coerce")
        raw_f = pd.to_numeric(row.get('VHI_기능', 0), errors="coerce")
        raw_e = pd.to_numeric(row.get('VHI_정서', 0), errors="coerce")
        raw_total = float(0 if pd.isna(raw_total) else raw_total)
        raw_p = float(0 if pd.isna(raw_p) else raw_p)
        raw_f = float(0 if pd.isna(raw_f) else raw_f)
        raw_e = float(0 if pd.isna(raw_e) else raw_e)

        # VHI는 UI에서 VHI-10(0~40) 기반으로 입력되므로,
        # training_data의 VHI-30(총점 0~120, 하위척도 0~40)을 VHI-10 스케일로 변환해 사용합니다.
        # UI에서 계산하는 분해(기능 0~20 / 신체 0~12 / 정서 0~8)와 동일하게 맞춥니다.
        if raw_total <= 40 and raw_f <= 20 and raw_p <= 12 and raw_e <= 8:
            vhi_total, vhi_p, vhi_f, vhi_e = raw_total, raw_p, raw_f, raw_e
        else:
            vhi_f = (raw_f / 40.0) * 20.0
            vhi_p = (raw_p / 40.0) * 12.0
            vhi_e = (raw_e / 40.0) * 8.0
            vhi_total = vhi_f + vhi_p + vhi_e

        sex_num = sex_to_num(row.get('성별', None))

        data_list.append([
            row.get('F0', 0), row.get('Range', 0), row.get('강도(dB)', 0), row.get('SPS', 0),
            vhi_total, vhi_p, vhi_f, vhi_e, sex_num,
            row.get('음도(청지각)', 0), row.get('음도범위(청지각)', 0), row.get('강도(청지각)', 0),
            row.get('말속도(청지각)', 0), row.get('조음정확도(청지각)', 0),
            diagnosis, subgroup
        ])

    df = pd.DataFrame(data_list, columns=FEATS_STEP2 + ['Diagnosis', 'Subgroup'])
    for col in FEATS_STEP2:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 결측 처리
    for col in ['F0', 'Range', 'Intensity', 'SPS',
                'P_Pitch', 'P_Range', 'P_Loudness', 'P_Rate', 'P_Artic', 'Sex']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())
    for col in ['VHI_Total', 'VHI_P', 'VHI_F', 'VHI_E']:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # Step1
    X1 = df[FEATS_STEP1].copy()
    y1 = df["Diagnosis"].astype(str).values
    model_step1 = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="lbfgs",
            max_iter=5000,
            class_weight="balanced",
            random_state=42
        ))
    ])
    model_step1.fit(X1, y1)

    # Step2 (PD 내부)
    df_pd = df[df["Diagnosis"] == "Parkinson"].copy()
    if df_pd.empty:
        return model_step1, None

    X2 = df_pd[FEATS_STEP2].copy()
    y2 = df_pd["Subgroup"].astype(str).values
    model_step2 = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", QuadraticDiscriminantAnalysis(reg_param=0.1))
    ])
    model_step2.fit(X2, y2)

    return model_step1, model_step2


try: model_step1, model_step2 = train_models()
except: model_step1, model_step2 = None, None

# training_data 기반 cut-off(확률 임계값) 자동 산출
try:
    _tf = "training_data.xlsx" if os.path.exists("training_data.xlsx") else "training_data.csv"
    _mt = os.path.getmtime(_tf) if os.path.exists(_tf) else None
    CUTS = compute_cutoffs_from_training(_mt)
except Exception:
    CUTS = None

# ==========================================
# [이메일 전송 함수] 파일명: 이름.wav
# ==========================================
def send_email_and_log_sheet(wav_path, patient_info, analysis, diagnosis):
    # Secrets가 없으면(또는 시트명이 없으면) 클라우드 전송 대신 SQLite에 저장
    if not globals().get("HAS_GCP_SECRETS", True) or (SHEET_NAME is None):
        try:
            step1_meta = st.session_state.get("save_ready_data", {}).get("step1_meta", st.session_state.get("step1_meta", {}))
        except Exception:
            step1_meta = {}
        try:
            save_to_sqlite(wav_path, patient_info, analysis, diagnosis, step1_meta)
            return True, "Secrets 미설정: 구글시트/이메일 대신 SQLite에 저장했습니다."
        except Exception as e:
            return False, f"Secrets 미설정 + SQLite 저장 실패: {e}"

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

        if not worksheet.row_values(1):
            worksheet.append_row([
                "Timestamp", "Filename", "Name", "Age", "Gender",
                "F0", "Range", "Intensity_dB", "SPS", 
                "VHI_Total", "VHI_P", "VHI_F", "VHI_E",
                "P_Artic", "P_Pitch", "P_Loud", "P_Rate", "P_PRange",
                "Final_Diagnosis", "Normal_Prob"
            ])
            
        row_data = [
            timestamp, log_filename,
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
        
        # [수정] 이메일 첨부 파일명: 이름.wav
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
        # 첨부 파일명 설정
        part.add_header("Content-Disposition", f"attachment; filename={email_attach_name}")
        msg.attach(part)

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())
        server.quit()

        # (선택) 클라우드 전송 성공 시에도 SQLite에 로그 저장
        try:
            step1_meta = st.session_state.get("save_ready_data", {}).get("step1_meta", st.session_state.get("step1_meta", {}))
            save_to_sqlite(wav_path, patient_info, analysis, diagnosis, step1_meta)
            return True, "메일/시트 저장 완료 + SQLite 로그 저장 완료"
        except Exception:
            return True, "메일 전송 및 시트 저장 완료"

    except Exception as e:
        return False, str(e)

# ==========================================
# [SMR 측정 함수]
# ==========================================
def auto_detect_smr_events(sound_path, top_n=20):
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
        return candidates, len(candidates)
    except:
        return [], 0

# ==========================================
# [분석 로직] Median Ratio 필터로 확실한 옥타브 제거
# ==========================================
def plot_pitch_contour_plotly(sound_path, f0_min, f0_max):
    try:
        sound = parselmouth.Sound(sound_path)
        pitch = call(sound, "To Pitch", 0.0, f0_min, f0_max)
        pitch_array = pitch.selected_array['frequency']
        pitch_values = np.array(pitch_array, dtype=np.float64)
        duration = sound.get_total_duration()
        n_points = len(pitch_values)
        time_array = np.linspace(0, duration, n_points)
        
        valid_indices = pitch_values != 0
        valid_times = time_array[valid_indices]
        valid_pitch = pitch_values[valid_indices]

        if len(valid_pitch) > 0:
            median_f0 = np.median(valid_pitch)
            lower_bound = median_f0 * 0.6
            upper_bound = median_f0 * 1.6
            
            clean_mask = (valid_pitch >= lower_bound) & (valid_pitch <= upper_bound)
            clean_p = valid_pitch[clean_mask]
            clean_t = valid_times[clean_mask]
            
            if len(clean_p) > 0:
                mean_f0 = np.mean(clean_p)
                rng = np.max(clean_p) - np.min(clean_p)
            else:
                mean_f0, rng = 0, 0
                clean_p, clean_t = [], []
        else:
            clean_p, clean_t = [], []
            mean_f0, rng = 0, 0

        fig = go.Figure()
        if len(clean_p) > 0:
            fig.add_trace(go.Scatter(x=clean_t, y=clean_p, mode='markers', marker=dict(size=4, color='red'), name='Pitch'))
            y_min = max(0, np.min(clean_p) - 20)
            y_max = np.max(clean_p) + 20
            fig.update_layout(title="음도 컨투어 (이상치 제거됨)", xaxis_title="Time(s)", yaxis_title="Hz", height=300, yaxis=dict(range=[y_min, y_max]))
        else:
            fig.update_layout(title="음도 컨투어 (감지된 음성 없음)", height=300)

        return fig, mean_f0, rng, duration
    except: return None, 0, 0, 0

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
    if artic >= 75: positives.append(f"청지각적 조음 정확도가 {artic}점으로 양호하여, 상대방이 말을 알아듣기에 명료한 상태입니다.")
    if sps < 4.5: positives.append(f"말속도가 {sps:.2f} SPS로 측정되었습니다. 파킨슨병에서 흔히 나타나는 급격한 가속 현상(Festination) 없이 안정적인 속도를 유지하고 있습니다.")
    if db >= 60: positives.append(f"평균 음성 강도가 {db:.1f} dB로, 일반적인 대화 수준(60dB 이상)의 성량을 튼튼하게 유지하고 있습니다.")

    if db < 60: negatives.append(f"평균 음성 강도가 {db:.1f} dB로 다소 작습니다. 이는 파킨슨병의 대표적 증상인 '강도 감소(Hypophonia)'와 유사하여 발성 훈련이 필요할 수 있습니다.")
    if sps >= 4.5: negatives.append(f"말속도가 {sps:.2f} SPS로 지나치게 빠릅니다. 이는 발화 제어가 어려워 말이 빠르지는 가속 징후(Short rushes of speech)일 가능성이 있습니다.")
    if artic < 70: negatives.append(f"청지각적 조음 정확도가 {artic}점으로 다소 낮습니다. 발음이 불분명해지는 조음 장애(Dysarthria) 징후가 관찰됩니다.")
    if vhi >= 20: negatives.append(f"VHI 총점이 {vhi}점으로 높습니다. 환자 스스로 음성 문제로 인한 생활의 불편함과 심리적 위축을 크게 느끼고 있습니다.")
    if vhi_e >= 5: negatives.append("특히 VHI 정서(E) 점수가 높아, 말하기에 대한 불안감이나 자신감 저하가 감지됩니다.")
    return positives, negatives

# --- UI Title ---
st.title("파킨슨병 환자 하위유형 분류 프로그램")
st.markdown("이 프로그램은 청지각적 평가, 음향학적 분석, 자가보고(VHI-10) 데이터를 통합하여 파킨슨병 환자의 음성 특성을 3가지 하위 유형으로 분류합니다.")

# 1. 사이드바
with st.sidebar:
    st.header("👤 대상자 정보 (필수)")
    subject_name = st.text_input("이름 (실명/ID)", "참여자")
    subject_age = st.number_input("나이", 1, 120, 60)
    subject_gender = st.selectbox("성별", ["M", "F"])

# 2. 데이터 수집
st.header("1. 음성 데이터 수집")
if 'user_syllables' not in st.session_state: st.session_state.user_syllables = 80
if 'source_type' not in st.session_state: st.session_state.source_type = None

col_rec, col_up = st.columns(2)
TEMP_FILENAME = "temp_for_analysis.wav"

with col_rec:
    st.markdown("#### 🎙️ 마이크 녹음")
    font_size = st.slider("🔍 글자 크기", 15, 50, 28, key="fs_read")
    
    # 문단 선택
    read_opt = st.radio("📖 낭독 문단 선택", ["1. 산책 (일반용 - 69음절)", "2. 바닷가의 추억 (SMR/정밀용 - 80음절)"])
    
    def styled_text(text, size): 
        return f"""<div style="font-size: {size}px; line-height: 1.8; border: 1px solid #ddd; padding: 15px; background-color: #f9f9f9; color: #333;">{text}</div>"""

    if "바닷가" in read_opt:
        read_text = "바닷가에 파도가 칩니다. 무지개 아래 바둑이가 뜁니다. 보트가 지나가고 버터구이를 먹습니다. 포토카드를 부탁해서 돋보기로 봅니다. 시장에서 빈대떡을 사 먹었습니다."
        default_syl = 80
    else:
        read_text = "높은 산에 올라가 맑은 공기를 마시며 소리를 지르면 가슴이 활짝 열리는 듯하다. 바닷가에 나가 조개를 주으며 넓게 펼쳐있는 바다를 바라보면 내 마음 역시 넓어지는 것 같다."
        default_syl = 69
        
    st.markdown(styled_text(read_text, font_size), unsafe_allow_html=True)
    
    # 음절 수 자동 변경
    syllables_rec = st.number_input("전체 음절 수", 1, 500, default_syl, key=f"syl_rec_{read_opt}")
    st.session_state.user_syllables = syllables_rec
    
    audio_buf = st.audio_input("낭독 녹음")
    if st.button("🎙️ 녹음된 음성 분석"):
        if audio_buf:
            with open(TEMP_FILENAME, "wb") as f: f.write(audio_buf.read())
            st.session_state.current_wav_path = os.path.join(os.getcwd(), TEMP_FILENAME)
            run_analysis_logic(st.session_state.current_wav_path)
        else: st.warning("녹음부터 해주세요.")

with col_up:
    st.markdown("#### 📂 파일 업로드")
    up_file = st.file_uploader("WAV 파일 선택", type=["wav"])
    if up_file: st.audio(up_file, format='audio/wav')
    if st.button("📂 업로드 파일 분석"):
        if up_file:
            with open(TEMP_FILENAME, "wb") as f: f.write(up_file.read())
            st.session_state.current_wav_path = os.path.join(os.getcwd(), TEMP_FILENAME)
            run_analysis_logic(st.session_state.current_wav_path)
        else: st.warning("파일을 올려주세요.")

# 3. 결과 및 저장
if st.session_state.get('is_analyzed'):
    st.markdown("---")
    st.subheader("2. 분석 결과 및 보정")
    
    c1, c2 = st.columns([2, 1])
    
    with c1: 
        st.plotly_chart(st.session_state['fig_plotly'], use_container_width=True)
    
    with c2:
        db_adj = st.slider("강도(dB) 보정", -50.0, 50.0, -10.0)
        final_db = st.session_state['mean_db'] + db_adj
        
        range_adj = st.slider("음도범위(Hz) 보정", 0.0, 300.0, float(st.session_state['pitch_range']))
        s_time, e_time = st.slider("말속도 구간(초)", 0.0, st.session_state['duration'], (0.0, st.session_state['duration']), 0.01)
        sel_dur = max(0.1, e_time - s_time)
        final_sps = st.session_state.user_syllables / sel_dur
        
        st.write("#### 📊 음향학적 분석 결과")
        result_df = pd.DataFrame({
            "항목": ["평균 강도(dB)", "평균 음도(Hz)", "음도 범위(Hz)", "말속도(SPS)"],
            "수치": [f"{final_db:.2f}", f"{st.session_state['f0_mean']:.2f}", f"{range_adj:.2f}", f"{final_sps:.2f}"]
        })
        st.dataframe(result_df, hide_index=True)

    st.markdown("---")
    if st.session_state.get('smr_events'):
        st.markdown("##### 🔎 SMR 자동 분석 (단어 매칭)")
        events = st.session_state['smr_events']
        smr_df_data = {}
        words = ["바닷가", "파도가", "무지개", "바둑이", "보트가", "버터구이", "포토카드", "부탁해", "돋보기", "빈대떡"]
        
        for i, word in enumerate(words):
            if i < len(events):
                ev = events[i]
                status = "🟢 양호" if ev['depth'] >= 20 else ("🟡 주의" if ev['depth'] >= 15 else "🔴 불량")
                val = f"{ev['depth']:.1f}dB\n{status}"
            else:
                val = "미감지"
            smr_df_data[word] = [val]
        
        st.dataframe(pd.DataFrame(smr_df_data), use_container_width=True)

    st.markdown("---")
    st.subheader("3. 청지각 및 VHI-10 입력")
    cc1, cc2 = st.columns([1, 1.2])
    with cc1:
        st.markdown("#### 🔊 청지각 평가")
        p_artic = st.slider("조음 정확도", 0, 100, 50)
        p_pitch = st.slider("음도", 0, 100, 50)
        p_prange = st.slider("음도 범위", 0, 100, 50)
        p_loud = st.slider("강도", 0, 100, 50)
        p_rate = st.slider("말속도", 0, 100, 50)
    with cc2:
        st.markdown("#### 📝 VHI-10")
        vhi_opts = [0, 1, 2, 3, 4]
        
        with st.expander("VHI-10 문항 입력 (클릭해서 펼치기)", expanded=True):
            q1 = st.select_slider("1. 사람들이 내 목소리를 듣는데 어려움을 느낀다.", options=vhi_opts)
            q2 = st.select_slider("2. 사람들이 내 말을 잘 못 알아들어 반복해야 한다.", options=vhi_opts)
            q3 = st.select_slider("3. 낯선 사람들과 전화로 대화하는 것이 어렵다.", options=vhi_opts)
            q4 = st.select_slider("4. 목소리 문제로 인해 긴장된다.", options=vhi_opts)
            q5 = st.select_slider("5. 목소리 문제로 인해 사람들을 피하게 된다.", options=vhi_opts)
            q6 = st.select_slider("6. 내 목소리 때문에 짜증이 난다.", options=vhi_opts)
            q7 = st.select_slider("7. 목소리 문제로 수입에 지장이 있다.", options=vhi_opts)
            q8 = st.select_slider("8. 내 목소리 문제로 대화가 제한된다.", options=vhi_opts)
            q9 = st.select_slider("9. 내 목소리 때문에 소외감을 느낀다.", options=vhi_opts)
            q10 = st.select_slider("10. 목소리를 내는 것이 힘들다.", options=vhi_opts)

        vhi_f = q1 + q2 + q5 + q7 + q8
        vhi_p = q3 + q4 + q6
        vhi_e = q9 + q10
        vhi_total = vhi_f + vhi_p + vhi_e
        
        st.markdown("##### 📊 영역별 점수")
        col_v1, col_v2, col_v3, col_v4 = st.columns(4)
        col_v1.metric("총점", f"{vhi_total}점")
        col_v2.metric("기능(F)", f"{vhi_f}점")
        col_v3.metric("신체(P)", f"{vhi_p}점")
        col_v4.metric("정서(E)", f"{vhi_e}점")

    st.markdown("---")
    st.subheader("4. 최종 진단 및 클라우드 전송")
    
    if st.button("🚀 진단 결과 확인", key="btn_diag"):
        if model_step1:
            # 성별 feature
            sex_num_ui = sex_to_num(subject_gender)

            # Step1: PD 확률 cut-off (training_data 기반)
            pd_cut = 0.5
            if CUTS and isinstance(CUTS, dict) and "step1_cutoff" in CUTS and CUTS["step1_cutoff"] is not None:
                pd_cut = float(CUTS["step1_cutoff"])

            # 기본값(저장용)
            p_pd = 0.0
            p_norm = 1.0

            # 조음정확도(p_artic) 78점 이상이면 Normal로 강제하던 규칙은 제거했습니다.
            if False:  # (removed rule)
                pass
                
                
            else:
                input_1 = pd.DataFrame([[
                    st.session_state['f0_mean'], range_adj, final_db, final_sps,
                    vhi_total, vhi_p, vhi_f, vhi_e,
                    sex_num_ui
                ]], columns=FEATS_STEP1)

                proba_1 = model_step1.predict_proba(input_1)[0]
                classes_1 = list(model_step1.classes_)
                if "Parkinson" in classes_1:
                    p_pd = float(proba_1[classes_1.index("Parkinson")])
                if "Normal" in classes_1:
                    p_norm = float(proba_1[classes_1.index("Normal")])
                else:
                    p_norm = 1.0 - p_pd

                prob_normal = p_norm * 100.0

                # cut-off 기준으로 판정
                if p_pd >= pd_cut:
                    st.error(f"🔴 **파킨슨 가능성 (PD) ({p_pd*100:.1f}%)**  | cut-off={pd_cut:.2f}")
                    if model_step2:
                        input_2 = pd.DataFrame([[
                            st.session_state['f0_mean'], range_adj, final_db, final_sps,
                            vhi_total, vhi_p, vhi_f, vhi_e,
                            sex_num_ui,
                            p_pitch, p_prange, p_loud, p_rate, p_artic
                        ]], columns=FEATS_STEP2)

                        probs_sub = model_step2.predict_proba(input_2)[0]
                        sub_classes = list(model_step2.classes_)
                        j = int(np.argmax(probs_sub))
                        pred_sub = sub_classes[j]
                        pred_prob = float(probs_sub[j])
                        final_decision = pred_sub

                        st.info(f"➡️ PD 하위 집단 예측: **{pred_sub}** ({pred_prob*100:.1f}%)")

                        # Step2 class별 cut-off (학습기반) - 미만이면 불확실 경고
                        sub_cut = None
                        if CUTS and isinstance(CUTS, dict):
                            sub_cut = (CUTS.get("step2_cutoff_by_class") or {}).get(pred_sub, None)
                        if sub_cut is not None and pred_prob < float(sub_cut):
                            st.warning(f"⚠️ 예측 확률이 학습기반 cut-off({float(sub_cut):.2f}) 미만입니다. '불확실'로 해석/재검 권고")
                            final_decision = f"{pred_sub} (불확실)"
                    else:
                        final_decision = "Parkinson"
                else:
                    st.success(f"🟢 **정상 음성 (Normal) ({prob_normal:.1f}%)**  | PD={p_pd*100:.1f}% , cut-off={pd_cut:.2f}")
                    final_decision = "Normal"

            # Step1 메타(저장/로그용)
            st.session_state.step1_meta = {"p_pd": p_pd, "p_normal": p_norm, "cutoff": pd_cut}

            # 해석 텍스트
            positives, negatives = generate_interpretation(prob_normal, final_db, final_sps, range_adj, p_artic, vhi_total, vhi_e)
            st.markdown("##### ✅ 정상일 확률이 높게 나온 이유")
            for p in positives: st.write(f"- {p}")
            st.markdown("##### ⚠️ 파킨슨 가능성이 존재하는 이유")
            for n in negatives: st.write(f"- {n}")


            # 저장/전송용 데이터 패키징
            st.session_state.save_ready_data = {
                'wav_path': st.session_state.current_wav_path,
                'patient': {'name': subject_name, 'age': subject_age, 'gender': subject_gender},
                'analysis': {
                    'f0': st.session_state['f0_mean'], 'range': range_adj, 'db': final_db, 'sps': final_sps,
                    'vhi_total': vhi_total, 'vhi_p': vhi_p, 'vhi_f': vhi_f, 'vhi_e': vhi_e,
                    'p_artic': p_artic, 'p_pitch': p_pitch, 'p_loud': p_loud, 'p_rate': p_rate, 'p_prange': p_prange
                },
                'diagnosis': {'final': final_decision, 'normal_prob': prob_normal},
                'step1_meta': st.session_state.get('step1_meta', {"p_pd": p_pd, "p_normal": p_norm, "cutoff": pd_cut})
            }
            st.session_state.is_saved = False

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
