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
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# -------------------------------
# Step1 screening 메시지(확률 구간별)
# -------------------------------
def step1_screening_band(p_pd: float, pd_cut: float = 0.50):
    """
    Step1(정상 vs PD) 스크리닝 확률(p_pd)에 따라 안내 문구/톤을 조절합니다.
    Return: (kind, headline, band_code)
      - kind: 'success'|'warning'|'error' (Streamlit 배너 색상)
      - headline: 사용자 안내 문구(스크리닝/추정 표현)
      - band_code: 후속 해석(섹션 제목/경계 안내)에 사용할 내부 코드
    """
    try:
        p_pd = float(p_pd)
    except Exception:
        p_pd = 0.0

    # 확률 구간(서비스/임상용 추천)
    if p_pd <= 0.10:
        return ("success", "정상 범위로 판단됩니다(정상 가능성이 매우 높음).", "normal_very_high")
    if p_pd < 0.30:
        return ("success", "정상 범위일 가능성이 높습니다.", "normal_high")

    # 0.30~0.40: 정상 쪽이 우세한 경계(혼재)로 보되, 과도한 경고를 피하기 위해 '정상 우세'로 안내
    if p_pd < 0.40:
        return ("success", "정상 가능성이 더 높습니다(일부 지표가 PD 학습군과 겹칠 수 있어 경계로 분류).", "border_mixed_normal_lean")

    # 0.40~0.45: 컷오프에 가까운 정상-우세 경계 구간(재측정/추적 권장)
    if p_pd < 0.45:
        return ("warning", "정상 가능성이 더 높지만 일부 지표가 PD 학습군과 겹칠 수 있습니다(경계).", "border_mixed_normal_lean2")
    if p_pd < 0.55:
        return ("warning", f"컷오프({pd_cut:.2f}) 근처의 경계 구간입니다(추가 평가/재측정 권장).", "border_cutoff")
    if p_pd < 0.70:
        return ("warning", "파킨슨병 관련 음성 특징이 관찰될 가능성이 있습니다.", "pd_possible")
    if p_pd < 0.90:
        return ("error", "파킨슨병 관련 음성 특징이 뚜렷할 가능성이 높습니다.", "pd_high")
    return ("error", "파킨슨병 관련 음성 특징이 매우 강하게 관찰됩니다.", "pd_very_high")



@st.cache_data
def get_step1_training_stats(_file_mtime=None):
    """
    Step1(정상 vs PD) 해석 보강용 통계(학습데이터 기준).
    - 중앙값(robust) 기반으로 입력값이 어느 집단에 더 가까운지 설명하기 위해 사용
    - 서비스 안정성: 모델/파이프라인 내부에서 계수 추출이 실패해도 설명이 '공란'이 되지 않도록 하는 안전장치
    """
    training_path = get_training_file()
    if training_path is None:
        return None

    try:
        df = pd.read_csv(training_path) if training_path.lower().endswith(".csv") else pd.read_excel(training_path)
    except Exception:
        return None

    label_col = "진단결과 (Label)"
    if label_col not in df.columns:
        return None

    labels = df[label_col].astype(str).str.lower()
    is_pd = labels.str.startswith("pd_")
    is_normal = labels.eq("normal")

    feats = ["F0", "Range", "강도(dB)", "SPS"]
    stats = {}
    for f in feats:
        if f not in df.columns:
            continue
        pd_vals = pd.to_numeric(df.loc[is_pd, f], errors="coerce").dropna()
        n_vals = pd.to_numeric(df.loc[is_normal, f], errors="coerce").dropna()
        if len(pd_vals) < 2 or len(n_vals) < 2:
            continue
        stats[f] = {
            "pd_med": float(pd_vals.median()),
            "n_med": float(n_vals.median()),
            "pd_mean": float(pd_vals.mean()),
            "n_mean": float(n_vals.mean()),
        }
    return stats if stats else None


def explain_step1_by_training(stats, x_dict, topk=3):
    """
    학습데이터(중앙값) 기준으로 입력값이 PD/정상 중 어디에 더 가까운지 설명 문장 생성.

    Return:
        reasons_normal: 정상 중앙값 쪽으로 더 가까운(또는 정상 쪽을 지지하는) 근거 TOP-K
        reasons_pd_strict: PD 중앙값 쪽으로 '명확히' 더 가까운 근거 TOP-K
        reasons_pd_closest: (경계/애매 구간 대비) PD 중앙값에 상대적으로 '가까운' 항목 TOP-K
            - 정상 중앙값이 더 가깝더라도, PD 중앙값과의 거리 기준으로 상위 항목을 반환
            - 임상용: PD 확률이 cut-off 근처일 때 "어떤 지표가 PD 학습군과 유사했는지"를 공란 없이 보여주기 위함
    """
    if not stats:
        return [], [], []

    reasons_pd_strict, reasons_n, reasons_pd_closest = [], [], []
    scored = []  # (abs_strength, strength, closeness_pd, f, x, pd_med, n_med)

    for f, s in stats.items():
        if f not in x_dict:
            continue
        try:
            x = float(x_dict[f])
        except Exception:
            continue
        if np.isnan(x):
            continue

        pd_med = s.get("pd_med")
        n_med = s.get("n_med")
        if pd_med is None or n_med is None:
            continue

        d_pd = abs(x - pd_med)
        d_n = abs(x - n_med)
        denom = abs(n_med - pd_med) + 1e-6

        # +면 PD쪽, -면 정상쪽 (중앙값 기준 상대적 가까움)
        strength = float((d_n - d_pd) / denom)
        abs_strength = abs(strength)

        # PD 중앙값과의 상대적 근접도(0~1 근사): 1에 가까울수록 PD 중앙값에 가까움
        closeness_pd = float(max(0.0, 1.0 - (d_pd / denom)))

        scored.append((abs_strength, strength, closeness_pd, f, x, pd_med, n_med))

    if not scored:
        return [], [], []

    scored.sort(reverse=True, key=lambda t: (t[0], t[2]))

    def _fmt(f, x, pd_med, n_med):
        if f == "강도(dB)":
            name, fmt = "평균 음성 강도", f"{x:.1f}dB"
            pd_fmt, n_fmt = f"{pd_med:.1f}dB", f"{n_med:.1f}dB"
        elif f == "Range":
            name, fmt = "음도 범위", f"{x:.1f}Hz"
            pd_fmt, n_fmt = f"{pd_med:.1f}Hz", f"{n_med:.1f}Hz"
        elif f == "F0":
            name, fmt = "평균 음도(F0)", f"{x:.1f}Hz"
            pd_fmt, n_fmt = f"{pd_med:.1f}Hz", f"{n_med:.1f}Hz"
        elif f == "SPS":
            name, fmt = "말속도(SPS)", f"{x:.2f}"
            pd_fmt, n_fmt = f"{pd_med:.2f}", f"{n_med:.2f}"
        else:
            name, fmt = f, f"{x:.3f}"
            pd_fmt, n_fmt = f"{pd_med:.3f}", f"{n_med:.3f}"
        return name, fmt, pd_fmt, n_fmt

    # 1) '명확히' 더 가까운 근거(방향 포함)
    for _, strength, closeness_pd, f, x, pd_med, n_med in scored:
        name, fmt, pd_fmt, n_fmt = _fmt(f, x, pd_med, n_med)

        if strength > 0 and len(reasons_pd_strict) < topk:
            reasons_pd_strict.append(f"{name}가 {fmt}로 **정상 중앙값({n_fmt})보다 PD 중앙값({pd_fmt})에 더 가깝습니다**.")
        elif strength < 0 and len(reasons_n) < topk:
            reasons_n.append(f"{name}가 {fmt}로 **PD 중앙값({pd_fmt})보다 정상 중앙값({n_fmt})에 더 가깝습니다**.")

        if len(reasons_pd_strict) >= topk and len(reasons_n) >= topk:
            break

    # 2) 경계용: PD 중앙값 '근접도' 상위 항목(방향 무관)
    scored_by_pd = sorted(scored, reverse=True, key=lambda t: t[2])
    for _, strength, closeness_pd, f, x, pd_med, n_med in scored_by_pd:
        if len(reasons_pd_closest) >= topk:
            break
        name, fmt, pd_fmt, n_fmt = _fmt(f, x, pd_med, n_med)
        reasons_pd_closest.append(
            f"{name}가 {fmt}이며, **PD 중앙값({pd_fmt})과의 거리가 비교적 가깝습니다**(정상 중앙값 {n_fmt})."
        )

    return reasons_n[:topk], reasons_pd_strict[:topk], reasons_pd_closest[:topk]

# --- 공통 유틸: 숫자 안전 변환 ---
def _safe_float(x, default=None):
    """Convert to float safely. Returns `default` on failure or missing/NaN."""
    try:
        if x is None:
            return default
        # pandas / numpy NaN 처리
        try:
            if pd.isna(x):
                return default
        except Exception:
            pass
        return float(x)
    except Exception:
        return default




# --- 페이지 기본 설정 ---
st.set_page_config(page_title="파킨슨병 환자 하위유형 분류 프로그램", layout="wide")


# --- 임상 보정: Step1 PD 확률을 청지각/VHI 정황으로 보정(오탐 완화) ---
def _calibrate_pd_probability(p_raw,
                             vhi_total=None,
                             p_artic=None,
                             p_rate=None,
                             p_loud=None,
                             intensity_db=None,
                             sps=None):
    """Post-hoc calibration for clinical stability.
    Returns (p_calibrated, notes). Does NOT change the underlying model,
    only adjusts odds based on strong normal/red-flag evidence."""
    if p_raw is None:
        return None, []
    try:
        p = float(p_raw)
    except Exception:
        return None, []
    # keep away from 0/1 for odds
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    odds = p / (1.0 - p)

    notes = []

    # --- normal evidence (downweight) ---
    if vhi_total is not None and vhi_total <= 3:
        odds *= 0.70
        notes.append("VHI 낮음(보정↓)")
    if p_artic is not None and p_artic >= 70:
        odds *= 0.70
        notes.append("조음 정확도 양호(보정↓)")
    if p_loud is not None and p_loud >= 70:
        odds *= 0.85
        notes.append("청지각 강도 양호(보정↓)")
    if intensity_db is not None and intensity_db >= 65:
        odds *= 0.90
        notes.append("음향 강도 양호(보정↓)")
    if sps is not None and sps <= 5.8:
        odds *= 0.95
        notes.append("말속도 안정(보정↓)")

    # --- red flags (upweight) ---
    if vhi_total is not None and vhi_total >= 20:
        odds *= 1.25
        notes.append("VHI 높음(보정↑)")
    if p_artic is not None and p_artic <= 40:
        odds *= 1.35
        notes.append("조음 저하(보정↑)")
    if p_loud is not None and p_loud <= 40:
        odds *= 1.20
        notes.append("청지각 강도 저하(보정↑)")
    if sps is not None and sps >= 5.8:
        odds *= 1.15
        notes.append("말속도 빠름(보정↑)")

    p_adj = odds / (1.0 + odds)
    p_adj = float(min(max(p_adj, 0.0), 1.0))
    return p_adj, notes


# ==========================================
# [설명(이유) 자동 생성: 상위 기여 변수 TOP-K]
# - 규칙 기반 설명이 비어있을 때, 모델의 선형 기여도(표준화된 값 × 계수)를 이용해
#   '왜 그렇게 나왔는지'를 최소 3개 항목으로 출력합니다.
# - 서비스 안정성 목적: 과도한 단정 대신 '모델 기준으로' 표현합니다.
# ==========================================

FEAT_LABELS_STEP1 = {
    "F0_Z": "평균 음도(F0, 성별 정규화)",
    "F0": "평균 음도(F0)",
    "Range": "음도 범위(range)",
    "RangeNorm": "음도 범위/평균음도(Range/F0)",
    "Intensity": "평균 음성 강도(dB)",
    "SPS": "말속도(SPS)",
}

FEAT_LABELS_STEP2 = {
    "Intensity": "평균 음성 강도(dB)",
    "SPS": "말속도(SPS)",
    "P_Loudness": "강도(청지각)",
    "P_Rate": "말속도(청지각)",
    "P_Artic": "조음정확도(청지각)"
}

def _get_pipeline_parts(pipeline):
    """Return (imputer, scaler, estimator) if present, else (None, None, pipeline)."""
    imputer = None
    scaler = None
    est = pipeline
    try:
        # sklearn Pipeline
        if hasattr(pipeline, "named_steps"):
            steps = pipeline.named_steps
            # common estimator step names we might use
            for key in ("clf", "logit", "lr", "lda", "qda", "model"):
                if key in steps:
                    est = steps[key]
                    break
            else:
                # fallback: last step
                est = list(steps.values())[-1]
            imputer = steps.get("imputer")
            scaler = steps.get("scaler")
    except Exception:
        pass
    return imputer, scaler, est

def top_contrib_linear_binary(pipeline, x_row, feat_names, pos_label="Parkinson", topk=3, exclude_feats=None, allow_sps=True, display_override=None):
    """Return top contributors for linear binary estimator.

    exclude_feats: iterable of feature names to skip (e.g., {"Sex"}).
    allow_sps: if False, suppress SPS(말속도) in explanation to avoid 과도한 강조(컷오프 근처에서만 노출).
    display_override: dict feature->string to show as input value.
    """
    if exclude_feats is None:
        exclude_feats = set()
    else:
        exclude_feats = set(exclude_feats)
    display_override = display_override or {}

    imputer, scaler, est = _get_pipeline_parts(pipeline)
    X = np.asarray(x_row, dtype=float).reshape(1, -1)
    if imputer is not None:
        X = imputer.transform(X)
    if scaler is not None:
        Xs = scaler.transform(X)
    else:
        Xs = X

    # binary logistic: coef_ shape (1, n_features)
    coef = getattr(est, "coef_", None)
    classes = list(getattr(est, "classes_", []))
    if coef is None or len(classes) < 2:
        return [], []

    try:
        pos_idx = classes.index(pos_label)
    except Exception:
        pos_idx = 1

    # for logistic regression, coef corresponds to class 1 vs 0 if binary; align with pos_label if possible
    w = coef[0]
    if classes[1] != pos_label:
        w = -w

    contrib = Xs[0] * w

    idx_sorted = np.argsort(np.abs(contrib))[::-1]
    pos, neg = [], []
    for i in idx_sorted:
        name = feat_names[i]
        if name in exclude_feats:
            continue
        if (name == "SPS") and (not allow_sps):
            continue

        label = FEAT_LABELS_STEP1.get(name, FEAT_LABELS_STEP2.get(name, name))

        # display value
        if name in display_override:
            val_str = str(display_override[name])
        else:
            try:
                v = float(np.asarray(x_row, dtype=float)[i])
                val_str = f"{v:.2f}" if np.isfinite(v) else ""
            except Exception:
                val_str = ""

        if contrib[i] >= 0 and len(pos) < topk:
            pos.append(f"{label}이(가) 모델에서 PD 확률을 높이는 방향으로 기여했습니다" + (f" (입력: {val_str})" if val_str else ""))
        elif contrib[i] < 0 and len(neg) < topk:
            neg.append(f"{label}이(가) 모델에서 정상 확률을 높이는 방향으로 기여했습니다" + (f" (입력: {val_str})" if val_str else ""))

        if len(pos) >= topk and len(neg) >= topk:
            break

    return pos, neg

def top_contrib_linear_multiclass(pipeline, x_row, feat_names, pred_class, topk=3):
    """Return reasons for predicted class for linear multiclass estimator (LDA)."""
    imputer, scaler, est = _get_pipeline_parts(pipeline)
    X = np.asarray(x_row, dtype=float).reshape(1, -1)
    if imputer is not None:
        X = imputer.transform(X)
    if scaler is not None:
        Xs = scaler.transform(X)
    else:
        Xs = X

    classes = list(getattr(est, "classes_", []))
    coef = getattr(est, "coef_", None)
    if coef is None or len(classes) == 0:
        return []

    try:
        cidx = classes.index(pred_class)
    except ValueError:
        cidx = int(np.argmax(getattr(est, "predict_proba", lambda z: np.zeros((1,len(classes))))(Xs)[0])) if len(classes) else 0

    w = coef[cidx] if coef.ndim == 2 else coef
    contrib = Xs[0] * w
    idx_sorted = np.argsort(np.abs(contrib))[::-1][:topk]
    reasons = []
    for i in idx_sorted:
        name = feat_names[i]
        label = FEAT_LABELS_STEP2.get(name, FEAT_LABELS_STEP1.get(name, name))
        val = float(np.asarray(x_row, dtype=float)[i]) if np.isfinite(np.asarray(x_row, dtype=float)[i]) else None
        reasons.append(f"{label}이(가) 이 집단 판정에 크게 기여했습니다" + (f" (입력: {val:.2f})" if val is not None else ""))
    return reasons

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
FEATS_STEP1 = ['F0_Z', 'Range', 'Intensity', 'SPS']
# Step2는 PD 하위집단 표본이 작아(특히 말속도 집단) 고차원 특성에 불안정합니다.
# 임상적으로 구분력이 큰 핵심 변수(강도/말속도/조음)만 사용합니다.
FEATS_STEP2 = ['Intensity', 'SPS', 'P_Loudness', 'P_Rate', 'P_Artic']

# Step2 하위집단: Top1–Top2 차이가 작으면(혼합 패턴) 혼합형으로 표시 (임상용)
MIX_MARGIN_P = 0.10  # 10%p
def sex_to_num(x):
    """성별을 숫자 feature로 변환: 남/M=1.0, 여/F=0.0, 그 외/결측=0.5"""
    if x is None:
        return 0.5
    s = str(x).strip().lower()
    if s in ["m", "male", "man", "남", "남성", "남자", "1"]:
        return 1.0
    if s in ["f", "female", "woman", "여", "여성", "여자", "0"]:
        return 0.0
    return 0.5

# ==========================================

# [training_data 위치 탐색]
# - Streamlit Cloud/Linux는 대소문자 구분 + 실행 경로가 달라질 수 있어
#   app.py(이 파일) 기준으로 training_data.*를 찾도록 합니다.
# ==========================================
MODEL_LOAD_ERROR = ""

def get_training_file():
    base = Path(__file__).resolve().parent
    # 우선순위: xlsx > csv (같은 폴더)
    candidates = [
        base / "training_data.xlsx",
        base / "training_data.csv",
        base / "Training_data.xlsx",
        base / "Training_data.csv",
        base / "TRAINING_DATA.xlsx",
        base / "TRAINING_DATA.csv",
    ]
    for p in candidates:
        if p.exists():
            return p

    # 혹시 하위 폴더에 있을 경우(마지막 안전장치)
    for p in base.rglob("training_data.csv"):
        return p
    for p in base.rglob("training_data.xlsx"):
        return p
    return None
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
    training_path = get_training_file()
    if training_path is None:
        global MODEL_LOAD_ERROR
        MODEL_LOAD_ERROR = "training_data.csv/xlsx 파일을 찾지 못했습니다. app.py와 같은 폴더(레포 루트)에 training_data.csv를 두고 커밋했는지 확인하세요."
        return None

    target_file = training_path

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

        # (안정성) Step1 가드에서 사용할 수 있도록 VHI 합계를 session_state에 저장
        try:
            st.session_state["vhi_total"] = float(vhi_total)
            st.session_state["vhi_f"] = float(vhi_f)
            st.session_state["vhi_p"] = float(vhi_p)
            st.session_state["vhi_e"] = float(vhi_e)
        except Exception:
            pass


        sex_num = sex_to_num(row.get('성별', None))

        data_list.append([
            row.get('F0', 0), row.get('Range', 0), row.get('강도(dB)', 0), row.get('SPS', 0),
            vhi_total, vhi_p, vhi_f, vhi_e, sex_num,
            row.get('음도(청지각)', 0), row.get('음도범위(청지각)', 0), row.get('강도(청지각)', 0),
            row.get('말속도(청지각)', 0), row.get('조음정확도(청지각)', 0),
            diagnosis, subgroup
        ])

    df = pd.DataFrame(data_list, columns=['F0','Range','Intensity','SPS','VHI_Total','VHI_P','VHI_F','VHI_E','Sex','P_Pitch','P_Range','P_Loudness','P_Rate','P_Artic','Diagnosis','Subgroup'])

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

    # Step1 보강: 평균음도(F0)는 성별에 따라 절대값 스케일이 크게 다릅니다.
    # 성별은 Step1 모델 입력에서 제외하는 대신, F0를 성별 기준 z-score로 정규화(F0_Z)하여 편향을 완화합니다.
    try:
        f0_series = pd.to_numeric(df.get('F0'), errors='coerce')
        sex_series = pd.to_numeric(df.get('Sex'), errors='coerce')

        f0_all = f0_series.dropna()
        all_mean = float(f0_all.mean()) if len(f0_all) > 0 else 0.0
        all_std = float(f0_all.std(ddof=0)) if len(f0_all) > 1 else 1.0
        if (not np.isfinite(all_std)) or all_std < 1e-6:
            all_std = 1.0

        params = {'ALL': {'mean': all_mean, 'std': all_std}}
        for sex_val, key in [(1.0, 'M'), (0.0, 'F')]:
            s = f0_series[sex_series == sex_val].dropna()
            if len(s) > 0:
                m_ = float(s.mean())
                sd_ = float(s.std(ddof=0)) if len(s) > 1 else all_std
                if (not np.isfinite(sd_)) or sd_ < 1e-6:
                    sd_ = all_std if all_std > 1e-6 else 1.0
                params[key] = {'mean': m_, 'std': sd_}
            else:
                params[key] = params['ALL']

        def _f0z(row_f0, row_sex):
            try:
                sx = float(row_sex)
            except Exception:
                sx = 0.5
            if sx >= 0.75:
                p = params.get('M', params['ALL'])
            elif sx <= 0.25:
                p = params.get('F', params['ALL'])
            else:
                p = params['ALL']
            try:
                return (float(row_f0) - p['mean']) / p['std']
            except Exception:
                return np.nan

        df['F0_Z'] = [_f0z(f, s) for f, s in zip(f0_series, sex_series)]
        st.session_state['f0_norm_params'] = params
    except Exception:
        df['F0_Z'] = np.nan
        st.session_state['f0_norm_params'] = {'ALL': {'mean': 0.0, 'std': 1.0}, 'M': {'mean': 0.0, 'std': 1.0}, 'F': {'mean': 0.0, 'std': 1.0}}

    X1 = df[FEATS_STEP1].copy()
    y1 = df["Diagnosis"].astype(str).values

    loo = LeaveOneOut()
    oof_pd = np.zeros(len(df), dtype=float)

    pipe1 = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
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

    y1_bin = (y1 == 'Parkinson').astype(int)
    step1_cutoff, step1_sens, step1_spec = _youden_cutoff(y1_bin, oof_pd)
    # (안정성 우선) 과도한 오탐을 막기 위해 cut-off 하한을 둡니다.
    step1_cutoff = float(max(step1_cutoff, 0.60))

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
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto'))
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
    step1_cm = confusion_matrix(y1_bin, y_pred1, labels=[0, 1])  # 0=Normal,1=PD

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
    """training_data로 Step1/Step2 모델을 학습합니다."""
    global MODEL_LOAD_ERROR

    training_path = get_training_file()
    if training_path is None:
        MODEL_LOAD_ERROR = "training_data.csv/xlsx 파일을 찾지 못했습니다."
        return None, None

    try:
        if str(training_path).lower().endswith(".xlsx"):
            raw = pd.read_excel(training_path)
        else:
            raw = pd.read_csv(training_path, encoding="utf-8-sig")
    except Exception as e:
        MODEL_LOAD_ERROR = f"training_data 로드 실패: {type(e).__name__}: {e}"
        return None, None

    if '진단결과 (Label)' not in raw.columns:
        MODEL_LOAD_ERROR = "training_data에 '진단결과 (Label)' 컬럼이 없습니다."
        return None, None

    def _label_to_diag_and_sub(lab: str):
        s = str(lab).strip().lower()
        if 'normal' in s:
            return "Normal", "Normal"
        if 'pd_intensity' in s:
            return "Parkinson", "강도 집단"
        if 'pd_rate' in s:
            return "Parkinson", "말속도 집단"
        if 'pd_articulation' in s or 'pd_artic' in s:
            return "Parkinson", "조음 집단"
        return None, None

    # ---------- Step1 ----------
    X1_rows, y1 = [], []
    for _, row in raw.iterrows():
        diag, _sub = _label_to_diag_and_sub(row.get('진단결과 (Label)', ''))
        if diag is None:
            continue

        raw_total = pd.to_numeric(row.get('VHI총점', np.nan), errors="coerce")
        raw_p = pd.to_numeric(row.get('VHI_신체', np.nan), errors="coerce")
        raw_f = pd.to_numeric(row.get('VHI_기능', np.nan), errors="coerce")
        raw_e = pd.to_numeric(row.get('VHI_정서', np.nan), errors="coerce")

        if (pd.notna(raw_total) and raw_total <= 40) and (pd.notna(raw_f) and raw_f <= 20) and (pd.notna(raw_p) and raw_p <= 12) and (pd.notna(raw_e) and raw_e <= 8):
            vhi_total, vhi_p, vhi_f, vhi_e = float(raw_total), float(raw_p), float(raw_f), float(raw_e)
        else:
            vhi_f = (0 if pd.isna(raw_f) else float(raw_f)) / 40.0 * 20.0
            vhi_p = (0 if pd.isna(raw_p) else float(raw_p)) / 40.0 * 12.0
            vhi_e = (0 if pd.isna(raw_e) else float(raw_e)) / 40.0 * 8.0
            vhi_total = vhi_f + vhi_p + vhi_e

        sex_num = sex_to_num(row.get('성별', None))

        X1_rows.append([
            row.get('F0', np.nan),
            row.get('Range', np.nan),
            row.get('강도(dB)', np.nan),
            row.get('SPS', np.nan),
            sex_num
        ])
        y1.append(diag)

    X1 = np.array(X1_rows, dtype=float)
    
    # Step1 학습 데이터 기반 기준값(가드/해석용)
    # Range는 성별/과제/무성구간 처리에 민감해 정상 오탐을 유발할 수 있어,
    # "정상 케이스 + 다른 지표 양호" 조건에서 중립값으로 대체하는 가드에 사용합니다.
    try:
        X1_arr = np.array(X1_rows, dtype=float)
        _range_all = X1_arr[:, 1]
        _sex_all = X1_arr[:, 4]
        median_range_all = float(np.nanmedian(_range_all))
        median_range_m = float(np.nanmedian(_range_all[_sex_all == 0])) if np.any(_sex_all == 0) else median_range_all
        median_range_f = float(np.nanmedian(_range_all[_sex_all == 1])) if np.any(_sex_all == 1) else median_range_all
        globals()['STATS_STEP1'] = {
            "median_range_all": median_range_all,
            "median_range_m": median_range_m,
            "median_range_f": median_range_f,
        }
    except Exception:
        globals()['STATS_STEP1'] = {"median_range_all": 100.0, "median_range_m": 90.0, "median_range_f": 120.0}

    y1 = np.array(y1, dtype=str)

    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    model_step1 = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=4000, class_weight="balanced"))
    ])
    model_step1.fit(X1, y1)

    # ---------- Step2 (PD only) ----------
    X2_rows, y2 = [], []
    for _, row in raw.iterrows():
        diag, sub = _label_to_diag_and_sub(row.get('진단결과 (Label)', ''))
        if diag != "Parkinson" or sub == "Normal":
            continue

        X2_rows.append([
            row.get('강도(dB)', np.nan),              # Intensity
            row.get('SPS', np.nan),                   # SPS
            row.get('강도(청지각)', np.nan),          # P_Loudness
            row.get('말속도(청지각)', np.nan),        # P_Rate
            row.get('조음정확도(청지각)', np.nan)     # P_Artic
        ])
        y2.append(sub)

    X2 = np.array(X2_rows, dtype=float)
    y2 = np.array(y2, dtype=object)

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    model_step2 = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto"))
    ])
    model_step2.fit(X2, y2)

    return model_step1, model_step2


try:
    model_step1, model_step2 = train_models()
except Exception as e:
    MODEL_LOAD_ERROR = f"모델 학습 중 예외: {type(e).__name__}: {e}"
    model_step1, model_step2 = None, None

# training_data 기반 cut-off(확률 임계값) 자동 산출
try:
    _tp = get_training_file()
    _mt = float(_tp.stat().st_mtime) if _tp is not None else None
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
                # Range는 max-min 대신 퍼센타일 기반(p95-p5)로 계산해 무성/이상치 영향 완화
                if len(clean_p) >= 10:
                    p5, p95 = np.percentile(clean_p, [5, 95])
                else:
                    p5, p95 = float(np.min(clean_p)), float(np.max(clean_p))
                rng = float(max(0.0, p95 - p5))
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

def run_analysis_logic(file_path, gender=None):
    try:
        g = (gender or "").strip().upper()
        # 성별에 따라 Pitch 탐지 범위를 조정(남성: 낮은 floor, 여성: 높은 floor)
        if g == "M":
            f0_min, f0_max = 60, 300
        elif g == "F":
            f0_min, f0_max = 100, 500
        else:
            f0_min, f0_max = 70, 500

        fig, f0, rng, dur = plot_pitch_contour_plotly(file_path, f0_min, f0_max)
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

def generate_interpretation(prob_normal, db, sps, range_val, artic, vhi, vhi_e, sex=None, p_pd=None, pd_cut=None):
    positives, negatives = [], []
    if vhi < 15: positives.append(f"환자 본인의 주관적 불편함(VHI {vhi}점)이 낮아, 일상 대화에 심리적/기능적 부담이 적은 상태입니다.")
    # (임상 안정성) 남성은 정상에서도 음도범위가 상대적으로 좁을 수 있어 기준을 완화합니다.
    try:
        _sex = (sex or "").strip().upper()
    except Exception:
        _sex = ""
    range_thr = 80 if _sex == "M" else 100
    if range_val >= range_thr: positives.append(f"음도 범위가 {range_val:.1f}Hz로 넓게 나타나, 목소리에 생동감이 있고 억양의 변화가 자연스럽습니다.")
    else:
        # 음도범위가 좁게 측정된 경우(특히 남성은 선천적으로 좁을 수 있음)
        negatives.append(f"음도 범위가 {range_val:.1f}Hz로 좁게 측정되었습니다. 억양 변화가 단조롭게 나타나는 패턴은 PD 학습군과 겹칠 수 있으나, 과제(짧은 문장/단음절)나 Pitch 추정 불안정에서도 발생할 수 있어 재측정이 권장됩니다.")
    if artic >= 75: positives.append(f"청지각적 조음 정확도가 {artic}점으로 양호하여, 상대방이 말을 알아듣기에 명료한 상태입니다.")
    if sps < 5.8: positives.append(f"말속도가 {sps:.2f} SPS로 측정되었습니다. 말속도는 안정적인 범위입니다.")
    if db >= 60: positives.append(f"평균 음성 강도가 {db:.1f} dB로, 일반적인 대화 수준(60dB 이상)의 성량을 튼튼하게 유지하고 있습니다.")

    if db < 60: negatives.append(f"평균 음성 강도가 {db:.1f} dB로 낮게 측정되었습니다(※ 마이크/거리/환경에 따라 절대값은 달라질 수 있으며, 본 도구의 모델 기준으로 낮은 편입니다). 이는 파킨슨병에서 흔한 강도 감소(Hypophonia) 패턴과 유사하여 발성 훈련이 필요할 수 있습니다.")
    # 말속도(SPS) fast는 정상에서도 흔하므로, 컷오프 근처에서만 경고 근거로 노출합니다.
    near_cut = True
    try:
        if (p_pd is not None) and (pd_cut is not None):
            near_cut = abs(float(p_pd) - float(pd_cut)) <= 0.08
    except Exception:
        near_cut = True
    if (sps >= 5.8) and near_cut:
        negatives.append(f"말속도가 {sps:.2f} SPS로 빠른 편입니다. 정상 성인에서도 빠른 말속도는 나타날 수 있으나, 컷오프 근처에서는 PD 학습군의 말속도/리듬 특징과 겹칠 수 있어 추가 확인이 필요합니다.")
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
            run_analysis_logic(st.session_state.current_wav_path, subject_gender)
        else: st.warning("녹음부터 해주세요.")

with col_up:
    st.markdown("#### 📂 파일 업로드")
    up_file = st.file_uploader("WAV 파일 선택", type=["wav"])
    if up_file: st.audio(up_file, format='audio/wav')
    if st.button("📂 업로드 파일 분석"):
        if up_file:
            with open(TEMP_FILENAME, "wb") as f: f.write(up_file.read())
            st.session_state.current_wav_path = os.path.join(os.getcwd(), TEMP_FILENAME)
            run_analysis_logic(st.session_state.current_wav_path, subject_gender)
        else: st.warning("파일을 올려주세요.")

# 3. 결과 및 저장
if st.session_state.get('is_analyzed'):
    st.markdown("---")
    st.subheader("2. 분석 결과 및 보정")
    
    c1, c2 = st.columns([2, 1])
    
    with c1: 
        st.plotly_chart(st.session_state['fig_plotly'], use_container_width=True)
    
    with c2:
        # 강도(dB) 보정:
        # - 기본은 0 dB(측정값 그대로)로 두고, 필요 시 임상가가 환경에 맞게 조정할 수 있도록 슬라이더를 제공합니다.
        # - (옵션) 특정 환경에서 강도가 과대 측정되는 경우에만 -5 dB 고정을 사용할 수 있습니다.
        INTENSITY_CORR_DB_DEFAULT = 0.0
        INTENSITY_CORR_DB_LOCK_VALUE = -5.0

        lock_db = st.checkbox(
            "강도 보정 고정(-5 dB) 사용(옵션)",
            value=False,
            key="lock_db_corr",
            help="녹음 강도가 과대 측정되는 환경에서만 -5 dB 고정을 사용하세요. 일반적으로는 0 dB(측정값 그대로)를 권장합니다."
        )

        if lock_db:
            st.slider(
                "강도(dB) 보정",
                -50.0, 50.0,
                INTENSITY_CORR_DB_LOCK_VALUE,
                0.5,
                disabled=True,
                key="db_adj_locked",
                help="고정 모드에서는 -5 dB가 적용됩니다."
            )
            db_adj = INTENSITY_CORR_DB_LOCK_VALUE
            # lock 모드에서는 다음 실행에서도 동일한 값이 기본으로 유지되도록 저장
            st.session_state["db_adj"] = db_adj
        else:
            db_adj = st.slider(
                "강도(dB) 보정",
                -50.0, 50.0,
                float(st.session_state.get("db_adj", INTENSITY_CORR_DB_DEFAULT)),
                0.5,
                key="db_adj",
                help="마이크/환경에 따라 dB가 달라질 수 있습니다. 기본은 0 dB(측정값 그대로)이며 필요 시 수동 조정 가능합니다."
            )

        final_db = st.session_state['mean_db'] + db_adj
        range_adj = st.slider("음도범위(Hz) 보정", 0.0, 300.0, float(st.session_state['pitch_range']))
        # 모델 입력용: Range(Hz) 정규화(Range/F0)로 성별/평균F0 스케일 영향을 완화
        _f0_for_norm = float(st.session_state.get('f0_mean', 0) or 0)
        range_norm_ui = float(range_adj) / max(_f0_for_norm, 1e-6)
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
        p_artic = st.slider("조음 정확도", 0, 100, int(st.session_state.get("p_artic", 50)), key="p_artic")
        p_pitch = st.slider("음도", 0, 100, int(st.session_state.get("p_pitch", 50)), key="p_pitch")
        p_prange = st.slider("음도 범위", 0, 100, int(st.session_state.get("p_prange", 50)), key="p_prange")
        p_loud = st.slider("강도", 0, 100, int(st.session_state.get("p_loud", 50)), key="p_loud")
        p_rate = st.slider("말속도", 0, 100, int(st.session_state.get("p_rate", 50)), key="p_rate")
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

            # cut-off 하한(임상 안정성): 과도한 오탐 방지
            try:
                pd_cut = max(float(pd_cut), 0.60)
            except Exception:
                pd_cut = 0.60

            # 기본값(저장용)
            p_pd = 0.0
            p_norm = 1.0

            # 조음정확도(p_artic) 78점 이상이면 Normal로 강제하던 규칙은 제거했습니다.
            if False:  # (removed rule)
                pass
                
                
            else:
                # Step1 입력 구성(안정성 가드 포함)
                f0_in = _safe_float(st.session_state.get('f0_mean'))
                pr_in = _safe_float(locals().get('range_adj', st.session_state.get('pitch_range')))
                db_in = _safe_float(final_db)
                sps_in = _safe_float(final_sps)

                # Range(음도범위)는 정상 발화에서도 과제/무성구간/추정 실패로 작게 나올 수 있어
                # '정상 정황이 강한 경우'에는 학습데이터 중앙값으로 대체하여 오탐을 줄입니다.
                try:
                    vhi_now = float(st.session_state.get('vhi_total', 0.0) or 0.0)
                except Exception:
                    vhi_now = 0.0
                try:
                    artic_now = float(st.session_state.get('p_artic', 0.0) or 0.0)
                except Exception:
                    artic_now = 0.0

                sex_is_m = str(subject_gender).strip().lower() in ['m', 'male', '남', '남성']
                sex_is_f = str(subject_gender).strip().lower() in ['f', 'female', '여', '여성']

                if pr_in is None or (isinstance(pr_in, float) and not np.isfinite(pr_in)):
                    pr_used = None
                else:
                    pr_used = float(pr_in)
                pr_raw = pr_used  # 원본 Range(Hz)

                # 정상 정황: VHI 낮음 + 청지각 조음 양호 + 강도/말속도 극단 아님
                normal_context = (vhi_now <= 3.0) and (artic_now >= 70.0) and (db_in is not None and db_in >= 65.0) and (sps_in is not None and sps_in <= 5.8)

                if normal_context and pr_used is not None:
                    # 성별별 중앙값 사용(없으면 전체 중앙값)
                    med_all = float(globals().get('STATS_STEP1', {}).get('median_range_all', 100.0))
                    med_m = float(globals().get('STATS_STEP1', {}).get('median_range_m', med_all))
                    med_f = float(globals().get('STATS_STEP1', {}).get('median_range_f', med_all))
                    med = med_m if sex_is_m else (med_f if sex_is_f else med_all)

                    # 너무 좁게 측정된 경우에만 중립값으로 대체
                    if sex_is_m and pr_used < 70.0:
                        pr_used = med
                    elif sex_is_f and pr_used < 90.0:
                        pr_used = med
                    elif (not sex_is_m and not sex_is_f) and pr_used < 80.0:
                        pr_used = med

                # 디버그/투명성: Step1에 실제로 사용된 Range 기록
                try:
                    st.session_state['step1_range_raw'] = pr_raw
                    st.session_state['step1_range_used'] = pr_used
                    st.session_state['step1_range_guard'] = bool(normal_context and pr_raw is not None and pr_used != pr_raw)
                except Exception:
                    pass

                # 성별은 모델 입력에서 제외하지만, F0는 성별 기준으로 정규화(F0_Z)합니다.
                params = st.session_state.get('f0_norm_params') or {'ALL': {'mean': 0.0, 'std': 1.0}, 'M': {'mean': 0.0, 'std': 1.0}, 'F': {'mean': 0.0, 'std': 1.0}}
                try:
                    if sex_num_ui >= 0.75:
                        p_ = params.get('M', params['ALL'])
                    elif sex_num_ui <= 0.25:
                        p_ = params.get('F', params['ALL'])
                    else:
                        p_ = params['ALL']
                    mu_ = float(p_.get('mean', 0.0))
                    sd_ = float(p_.get('std', 1.0))
                    if (not np.isfinite(sd_)) or sd_ < 1e-6:
                        sd_ = 1.0
                    f0_z_used = (float(f0_in) - mu_) / sd_
                except Exception:
                    f0_z_used = np.nan
                st.session_state['step1_f0_z_used'] = f0_z_used

                input_1 = pd.DataFrame([[
                    f0_z_used, pr_used, db_in, sps_in
                ]], columns=FEATS_STEP1)

                proba_1 = model_step1.predict_proba(input_1.to_numpy())[0]
                classes_1 = list(model_step1.classes_)
                if "Parkinson" in classes_1:
                    p_pd = float(proba_1[classes_1.index("Parkinson")])
                if "Normal" in classes_1:
                    p_norm = float(proba_1[classes_1.index("Normal")])
                else:
                    p_norm = 1.0 - p_pd

                prob_normal = p_norm * 100.0

                # (임상 안정성) 남성에서 음도 범위가 선천적으로 좁을 수 있어,
                # 컷오프 근처에서는 Range 단독 신호가 과대 반영되지 않도록 p_pd를 아주 소폭 완화합니다.
                try:
                    if (subject_gender == "M") and (range_adj < 90) and (p_pd < (pd_cut + 0.07)):
                        p_pd = max(0.0, p_pd - 0.07)
                        prob_normal = (1.0 - p_pd) * 100.0
                except Exception:
                    pass
                # --- Step1 임상 보정(오탐 완화): 청지각/VHI 등 강한 정상 정황이면 PD odds를 낮춰 해석 안정화 ---
                try:
                    p_pd_raw = float(p_pd)
                except Exception:
                    p_pd_raw = None

                vhi_total = _safe_float(st.session_state.get("vhi_total"), default=None)
                p_artic = _safe_float(st.session_state.get("p_artic"), default=None)
                p_rate  = _safe_float(st.session_state.get("p_rate"), default=None)
                p_loud  = _safe_float(st.session_state.get("p_loud"), default=None)

                p_pd_cal, cal_notes = _calibrate_pd_probability(
                    p_pd_raw,
                    vhi_total=vhi_total,
                    p_artic=p_artic,
                    p_rate=p_rate,
                    p_loud=p_loud,
                    intensity_db=db_in,
                    sps=sps_in,
                )

                # 보정값이 계산되면 판정/밴드는 보정 확률로, 원확률은 참고로 저장/표시
                if p_pd_cal is not None:
                    st.session_state["step1_p_pd_raw"] = p_pd_raw
                    st.session_state["step1_p_pd_cal"] = p_pd_cal
                    st.session_state["step1_cal_notes"] = cal_notes
                    p_pd = float(p_pd_cal)
                    p_norm = 1.0 - p_pd
                    prob_normal = p_norm * 100.0
                else:
                    st.session_state["step1_p_pd_raw"] = p_pd_raw
                    st.session_state["step1_p_pd_cal"] = None
                    st.session_state["step1_cal_notes"] = []


                # cut-off 기준으로 판정
                if p_pd >= pd_cut:
                    kind, headline, band_code = step1_screening_band(p_pd, pd_cut)
                    if kind == "error":
                        st.error(f"🔴 **{headline}**  | Normal={prob_normal:.1f}%  PD={p_pd*100:.1f}% (cut-off={pd_cut:.2f})")
                    elif kind == "warning":
                        st.warning(f"🟡 **{headline}**  | Normal={prob_normal:.1f}%  PD={p_pd*100:.1f}% (cut-off={pd_cut:.2f})")
                    else:
                        st.success(f"🟢 **{headline}**  | Normal={prob_normal:.1f}%  PD={p_pd*100:.1f}% (cut-off={pd_cut:.2f})")
                    # (참고) 모델 원확률 vs 임상 보정 확률 표시
                    try:
                        p_raw_show = st.session_state.get("step1_p_pd_raw", None)
                        p_cal_show = st.session_state.get("step1_p_pd_cal", None)
                        notes_show = st.session_state.get("step1_cal_notes", [])
                        if p_raw_show is not None and p_cal_show is not None and abs(float(p_raw_show) - float(p_cal_show)) > 1e-6:
                            note_txt = ", ".join(list(notes_show)[:4]) if notes_show else ""
                            st.caption(f"모델 원확률: PD={float(p_raw_show)*100:.1f}% → 임상 보정 후: PD={float(p_cal_show)*100:.1f}% {(' | ' + note_txt) if note_txt else ''}")
                    except Exception:
                        pass


                    st.session_state.step1_band_code = band_code
                    if model_step2:
                        # Step2 입력(feature 축소 버전) — FEATS_STEP2에 맞춰 값만 구성
                        feat_map2 = {
                            'Intensity': final_db,
                            'SPS': final_sps,
                            'P_Loudness': p_loud,
                            'P_Rate': p_rate,
                            'P_Artic': p_artic,
                        }
                        input_2 = pd.DataFrame([[feat_map2.get(c, None) for c in FEATS_STEP2]], columns=FEATS_STEP2)

                        probs_sub = model_step2.predict_proba(input_2.to_numpy())[0]
                        sub_classes = list(model_step2.classes_)
                        j = int(np.argmax(probs_sub))
                        pred_sub = sub_classes[j]
                        pred_prob = float(probs_sub[j])
                        final_decision = pred_sub  # (안정성) 모델 예측 라벨은 기본적으로 유지

                        # --- 혼합형(Top1–Top2 차이 < MIX_MARGIN_P) 표시용 요약 ---
                        try:
                            _pairs = sorted(zip(sub_classes, probs_sub), key=lambda x: float(x[1]), reverse=True)
                            _top1_lbl, _top1_p = _pairs[0][0], float(_pairs[0][1])
                            _top2_lbl, _top2_p = (_pairs[1][0], float(_pairs[1][1])) if len(_pairs) > 1 else (None, 0.0)
                            _is_mixed = (_top2_lbl is not None) and ((_top1_p - _top2_p) < MIX_MARGIN_P)
                        except Exception:
                            _top1_lbl, _top1_p, _top2_lbl, _top2_p, _is_mixed = pred_sub, float(pred_prob), None, 0.0, False

                        # --- 표시/하이브리드 보정용 확률/청지각 점수(안전 파싱) ---
                        # Step2 클래스 확률을 라벨→확률 dict로 정리(이후 문구/혼합형 판단에 사용)
                        probs_map = {str(lbl): float(p) for lbl, p in zip(sub_classes, probs_sub)}

                        # (안정성) 청지각 점수는 session_state에서 우선 읽고, 없으면 로컬 변수로 fallback
                        percep_rate_score  = _safe_float(st.session_state.get("p_rate", locals().get("p_rate")))
                        percep_artic_score = _safe_float(st.session_state.get("p_artic", locals().get("p_artic")))

                        # (안정성) 하위집단 확률
                        intensity_prob = _safe_float(probs_map.get("강도 집단"))
                        rate_prob      = _safe_float(probs_map.get("말속도 집단"))
                        jo_prob        = _safe_float(probs_map.get("조음 집단"))

                        # --- 말속도→조음 보정(임상 우선): 조음이 낮고(≤40), 속도 청지각 신호가 높지 않으면(≤60) 조음 동반 가능 ---
                        rule_artic_soft = (percep_artic_score is not None) and (percep_artic_score <= 40) and ((percep_rate_score is None) or (percep_rate_score <= 60))
                        rule_artic_primary = (percep_artic_score is not None) and (percep_artic_score <= 25) and ((percep_rate_score is None) or (percep_rate_score <= 60))
                        hybrid_overrode_rate_to_artic = False
                        
                        # (중요) 강도 우세 케이스는 라벨을 뒤집지 않음. 말속도 우세 케이스에서만 '조음 우선' 보정을 허용.
                        if (pred_sub == "말속도 집단") and rule_artic_primary and (jo_prob is not None) and (rate_prob is not None):
                            hybrid_overrode_rate_to_artic = True
                        
                        # --- 결과 문구(혼합형/보정 포함) ---
                        if _is_mixed and (_top2_lbl is not None) and (not hybrid_overrode_rate_to_artic):
                            st.info(f"➡️ PD 하위 집단 예측 : 혼합형으로 {_top1_lbl}에 포함될 가능성이 더 높으며, {_top2_lbl} 문제를 동반할 수 있습니다({_top1_lbl} {_top1_p*100:.1f}%, {_top2_lbl} {_top2_p*100:.1f}%).")
                        elif hybrid_overrode_rate_to_artic:
                            st.info(f"➡️ PD 하위 집단 예측 : 혼합형으로 조음 집단에 포함될 가능성을 우선 고려합니다(말속도 집단 {rate_prob*100:.1f}%, 조음 집단 {jo_prob*100:.1f}%).")
                            st.info("🧩 하이브리드 분석 결과 조음 집단에 포함될 가능성이 더 높으며, 말속도 신호가 동반될 수 있습니다(하이브리드 보정).")
                        else:
                            st.info(f"➡️ PD 하위 집단 예측: **{pred_sub}** ({pred_prob*100:.1f}%)")
                        
                        # --- Hybrid 신호(설명 문구만; 강제 보정은 혼합형 문구로 대체) ---
                        # 기존 규칙: 조음 정확도 저하(≤40) + 속도 신호 높지 않음 → 조음 저하 동반 가능
                        if (pred_sub == "강도 집단") and rule_artic_soft and (not _is_mixed):
                            st.info("🧩 하이브리드 분석 결과 강도 집단에 포함될 가능성이 더 높으며, 조음 동반 가능성(혼합형)이 있습니다.")
                        elif (pred_sub == "말속도 집단") and rule_artic_soft and (not hybrid_overrode_rate_to_artic) and (jo_prob is not None) and (rate_prob is not None):
                            st.info(f"🧩 하이브리드 분석 결과 말속도 집단에 포함될 가능성이 더 높지만, 조음 문제가 동반될 수 있습니다(말속도 집단 {rate_prob*100:.1f}%, 조음 집단 {jo_prob*100:.1f}%).")
                        
                        else:
                            st.info("ℹ️ 임상 참고: PD 하위집단(강도/말속도/조음)은 **PD 데이터로만 학습**된 추정 결과입니다. 정상 케이스에서는 참고용으로만 해석하세요.")
                        
                        # ---- Spider/Radar chart: PD 하위집단 확률 시각화 (원래 UI 복원) ----
                        try:
                            labels = sub_classes
                            labels_with_probs = [f"{label}\n({prob*100:.1f}%)" for label, prob in zip(labels, probs_sub)]
                            fig_radar = plt.figure(figsize=(3, 3))
                            ax = fig_radar.add_subplot(111, polar=True)
                            angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
                            angles += angles[:1]
                            stats = probs_sub.tolist() + [probs_sub[0]]
                            ax.plot(angles, stats, linewidth=2, linestyle='solid', color='red')
                            ax.fill(angles, stats, 'red', alpha=0.25)
                            ax.set_xticks(angles[:-1])
                            ax.set_xticklabels(labels_with_probs)

                            c_chart, c_desc = st.columns([1, 2])
                            with c_chart:
                                st.pyplot(fig_radar)

                            with c_desc:
                                if "강도" in pred_sub:
                                    st.info("💡 특징: 목소리 크기가 작고 약합니다. (Hypophonia)")
                                elif "말속도" in pred_sub:
                                    st.info("💡 특징: 말이 빠르거나 리듬이 불규칙할 수 있습니다. (Rate/Rhythm)")
                                else:
                                    st.info("💡 특징: 발음이 뭉개지고 정확도가 떨어질 수 있습니다. (Articulation)")

                                with st.expander("📊 하위집단 확률(상세)", expanded=False):
                                    dfp = pd.DataFrame({
                                        "집단": labels,
                                        "확률(%)": (np.array(probs_sub) * 100).round(1)
                                    }).sort_values("확률(%)", ascending=False)
                                    
                                    # --- [설명 보강] 하위집단 분류에 기여한 상위 변수 TOP-3 ---
                                    try:
                                        x2_row = [final_db, final_sps, p_loud, p_rate, p_artic]
                                        contrib2 = top_contrib_linear_multiclass(model_step2, x2_row, FEATS_STEP2, pred_sub, topk=3)
                                        if contrib2:
                                            st.markdown("**🔎 이 하위집단 판정에 크게 기여한 요소(Top 3)**")
                                            for r in contrib2:
                                                st.write(f"- {r}")
                                    except Exception:
                                        pass

                                    st.dataframe(dfp, hide_index=True, use_container_width=True)
                        except Exception as e:
                            st.warning(f"레이더 차트 생성 실패: {e}")


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
                    # 확률 기반으로는 정상으로 분류되었더라도, 청지각/자가보고/음향 일부 지표에서 뚜렷한 이상 소견이 있으면
                    # 서비스 안정성을 위해 '정상(주의)'로 표시하고 추가 평가/추적검사를 권장합니다.
                    red_flags = []
                    try:
                        if p_artic is not None and float(p_artic) <= 40:
                            red_flags.append("조음정확도(청지각) ≤ 40")
                    except Exception:
                        pass
                    try:
                        if final_db is not None and float(final_db) <= 58:
                            red_flags.append("평균 음성 강도(dB) 낮음")
                    except Exception:
                        pass
                    try:
                        if (p_pd >= (pd_cut - 0.10)) and (final_sps is not None) and (float(final_sps) >= 5.8):
                            red_flags.append("말속도(SPS) 빠름(≥5.8, 컷오프 근처)")
                    except Exception:
                        pass
                    try:
                        if vhi_total is not None and float(vhi_total) >= 10:
                            red_flags.append("VHI-10 높음(≥10)")
                    except Exception:
                        pass

                    if red_flags:
                        kind, headline, band_code = step1_screening_band(p_pd, pd_cut)
                        # 정상으로 분류되었지만 red-flag가 있을 때는 경고로 고정
                        st.warning(
                            f"🟡 **정상(주의): {headline}**  | Normal={prob_normal:.1f}%  PD={p_pd*100:.1f}% (cut-off={pd_cut:.2f})"
                        )
                        st.write("관찰된 항목: " + ", ".join(red_flags))
                        final_decision = "Normal (주의)"
                    else:
                        kind, headline, band_code = step1_screening_band(p_pd, pd_cut)
                        # red-flag가 없으면 구간별 메시지 그대로 사용
                        if kind == "warning":
                            st.warning(f"🟡 **{headline}**  | Normal={prob_normal:.1f}%  PD={p_pd*100:.1f}% (cut-off={pd_cut:.2f})")
                        elif kind == "error":
                            st.error(f"🔴 **{headline}**  | Normal={prob_normal:.1f}%  PD={p_pd*100:.1f}% (cut-off={pd_cut:.2f})")
                        else:
                            st.success(f"🟢 **{headline}**  | Normal={prob_normal:.1f}%  PD={p_pd*100:.1f}% (cut-off={pd_cut:.2f})")
                        final_decision = "Normal"
            # --- (임상용) 정상(주의)에서도 PD 하위집단 추정 결과 표시(참고) ---
            # Step2 모델은 파킨슨 환자 데이터로 학습되어 '정상'에서의 해석에는 제한이 있습니다.
            show_step2_reference = False
            try:
                # 임상용: 정상/정상(주의)에서도 하위집단 추정(참고) 표시
                show_step2_reference = str(final_decision).startswith('Normal')
            except Exception:
                show_step2_reference = False

            if show_step2_reference and model_step2:
                st.info("ℹ️ **임상 참고:** PD 하위집단(강도/말속도/조음) 추정 결과입니다. (Step2는 PD 데이터로 학습되어 정상 케이스에서는 참고용으로만 해석하세요.)")
                try:
                    feat_map2 = {
                        'Intensity': final_db,
                        'SPS': final_sps,
                        'P_Loudness': p_loud,
                        'P_Rate': p_rate,
                        'P_Artic': p_artic,
                    }
                    input_2_ref = pd.DataFrame([[feat_map2.get(c, None) for c in FEATS_STEP2]], columns=FEATS_STEP2)

                    probs_sub_ref = model_step2.predict_proba(input_2_ref.to_numpy())[0]
                    sub_classes_ref = list(model_step2.classes_)
                    j_ref = int(np.argmax(probs_sub_ref))
                    pred_sub_ref = sub_classes_ref[j_ref]
                    pred_prob_ref = float(probs_sub_ref[j_ref])

                    # Hybrid rule + intensity guard (참고용 추정에도 동일 적용)
                    pred_sub_ref_final = pred_sub_ref  # 라벨 보정 없음(임상 안정성)

                    # --- 혼합형(Top1–Top2 차이 < MIX_MARGIN_P) 표시용 요약(참고) ---
                    try:
                        _pairs_r = sorted(zip(sub_classes_ref, probs_sub_ref), key=lambda x: float(x[1]), reverse=True)
                        _top1_lbl_r, _top1_p_r = _pairs_r[0][0], float(_pairs_r[0][1])
                        _top2_lbl_r, _top2_p_r = (_pairs_r[1][0], float(_pairs_r[1][1])) if len(_pairs_r) > 1 else (None, 0.0)
                        _is_mixed_r = (_top2_lbl_r is not None) and ((_top1_p_r - _top2_p_r) < MIX_MARGIN_P)
                    except Exception:
                        _top1_lbl_r, _top1_p_r, _top2_lbl_r, _top2_p_r, _is_mixed_r = pred_sub_ref, float(pred_prob_ref), None, 0.0, False

                    pred_sub_ref_display = pred_sub_ref
                    if _is_mixed_r:
                        pred_sub_ref_display = f"혼합형({_top1_lbl_r} 우세, {_top2_lbl_r} 동반)"

                    if _is_mixed_r and (_top2_lbl_r is not None):
                        if (_top1_lbl_r == "강도 집단") and (_top2_lbl_r == "조음 집단"):
                            st.info(f"➡️ PD 하위 집단 예측(참고) : **혼합형**으로 강도 집단에 포함될 가능성이 더 높으며, 조음 저하를 동반할 수 있습니다(강도 집단 {_top1_p_r*100:.1f}%, 조음 집단 {_top2_p_r*100:.1f}%).")
                        else:
                            st.info(f"➡️ PD 하위 집단 예측(참고): **혼합형({_top1_lbl_r} 우세, {_top2_lbl_r} 동반)** (Top1: {_top1_lbl_r} {_top1_p_r*100:.1f}%, Top2: {_top2_lbl_r} {_top2_p_r*100:.1f}%).")
                    else:
                        st.info(f"➡️ PD 하위 집단 예측(참고): **{pred_sub_ref_final}** ({pred_prob_ref*100:.1f}%)")

                    # --- Hybrid 신호(참고): 라벨 보정은 하지 않고, 동반 가능성만 안내 ---
                    intensity_prob_ref = float(probs_sub_ref[list(sub_classes_ref).index("강도 집단")]) if "강도 집단" in sub_classes_ref else None
                    jo_prob_ref = float(probs_sub_ref[list(sub_classes_ref).index("조음 집단")]) if "조음 집단" in sub_classes_ref else None
                    rate_prob_ref = float(probs_sub_ref[list(sub_classes_ref).index("말속도 집단")]) if "말속도 집단" in sub_classes_ref else None

                    percep_artic_score_ref = float(p_artic) if 'p_artic' in locals() and p_artic is not None else None

                    rule_artic_ref = (percep_artic_score_ref is not None) and (percep_artic_score_ref <= 40) and ((rate_prob_ref is None) or (rate_prob_ref < 0.45))
                    if rule_artic_ref and pred_sub_ref == "강도 집단" and jo_prob_ref is not None and jo_prob_ref > 0:
                        st.info("🧩 하이브리드 분석 결과(참고) 강도 집단에 포함될 가능성이 더 높으며, 조음 동반 가능성(혼합형)이 있습니다. (라벨 보정 없음)")
                    elif rule_artic_ref:
                        st.info("🧩 하이브리드 분석 결과(참고) 조음 저하 동반 가능성이 있습니다. (라벨 보정 없음)")
                    elif pred_sub_ref == "강도 집단" and jo_prob_ref is not None and _top2_lbl_r == "조음 집단" and ((_top1_p_r - _top2_p_r) < MIX_MARGIN_P):
                        st.info(f"🧩 혼합 패턴(참고): 강도({_top1_p_r*100:.1f}%) 우세이나 조음({_top2_p_r*100:.1f}%)도 근접합니다 → **혼합형(강도 우세, 조음 동반)** 으로 해석하세요.")
                    # Radar chart
                    try:
                        labels = sub_classes_ref
                        labels_with_probs = [f"{label}\n({prob*100:.1f}%)" for label, prob in zip(labels, probs_sub_ref)]
                        fig_radar_ref = plt.figure(figsize=(3, 3))
                        ax = fig_radar_ref.add_subplot(111, polar=True)
                        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
                        angles += angles[:1]
                        stats = probs_sub_ref.tolist() + [probs_sub_ref[0]]
                        ax.plot(angles, stats, linewidth=2, linestyle='solid', color='red')
                        ax.fill(angles, stats, 'red', alpha=0.25)
                        ax.set_thetagrids(np.degrees(angles[:-1]), labels_with_probs, fontsize=10)
                        ax.set_ylim(0, 1)
                        ax.grid(True)
                        st.pyplot(fig_radar_ref, use_container_width=False)
                        plt.close(fig_radar_ref)
                    except Exception:
                        pass
                except Exception:
                    st.info("참고용 하위집단 추정 중 오류가 발생했습니다.")

            # Step1 메타(저장/로그용)
            st.session_state.step1_meta = {"p_pd": p_pd, "p_normal": p_norm, "cutoff": pd_cut}
            # Range 가드가 적용되었는지(오탐 방지) 사용자에게 투명하게 표시
            try:
                if st.session_state.get('step1_range_guard'):
                    raw_r = st.session_state.get('step1_range_raw')
                    used_r = st.session_state.get('step1_range_used')
                    st.info(f"⚙️ 음도범위가 정상 정황에서 과도하게 좁게 측정되어(원본 {raw_r:.1f}Hz) 모델 입력은 중립값({used_r:.1f}Hz)으로 보정했습니다. (오탐 방지)")
            except Exception:
                pass

            # 해석 텍스트
            st.caption('※ 자가보고(VHI)는 **판정 확률 계산에는 사용하지 않고**, 해석/경고를 위한 참고 지표로만 표시됩니다.')
            positives, negatives = generate_interpretation(prob_normal, final_db, final_sps, range_adj, p_artic, vhi_total, vhi_e, sex=subject_gender, p_pd=p_pd, pd_cut=pd_cut)

            # --- [설명 보강] 규칙 기반 설명이 비어있을 때: 모델 TOP 기여 변수로 최소 3개 생성 ---
            # --- 자동 설명(모델 기여도): 실패해도 이유가 비지 않도록 ---
            x1_row = [
                st.session_state.get('step1_f0_z_used', np.nan),
                st.session_state.get('step1_range_used', range_adj if 'range_adj' in locals() else st.session_state.get('pitch_range')),
                final_db,
                final_sps,
            ]

            # SPS 근거는 cut-off 근처에서만 과도하게 강조되지 않도록 제한
            near_cutoff_sps = True
            try:
                near_cutoff_sps = abs(float(p_pd) - float(pd_cut)) <= 0.08
            except Exception:
                near_cutoff_sps = True

            disp_map = {
                'F0_Z': f"{float(f0_in):.2f}Hz (z={float(st.session_state.get('step1_f0_z_used', np.nan)):.2f})" if f0_in is not None else '',
                'Range': f"{float(pr_used):.2f}" if pr_used is not None else '',
                'Intensity': f"{float(final_db):.2f}" if final_db is not None else '',
                'SPS': f"{float(final_sps):.2f}" if final_sps is not None else '',
            }

            try:
                pos_auto, neg_auto = top_contrib_linear_binary(
                    model_step1,
                    x1_row,
                    FEATS_STEP1,
                    pos_label="Parkinson",
                    topk=3,
                    exclude_feats={"Sex"},
                    allow_sps=near_cutoff_sps,
                    display_override=disp_map,
                )
                # 정상 확률 설명이 비면(또는 너무 짧으면) 자동 설명을 섞어줌
                if not positives or len(positives) < 1:
                    positives = (positives or []) + (neg_auto[:3] if neg_auto else [])
                # PD 가능성 이유가 비면 자동 설명(=PD 쪽 기여) 추가
                if not negatives or len(negatives) < 1:
                    negatives = (negatives or []) + (pos_auto[:3] if pos_auto else [])
            except Exception:
                # 자동 설명이 실패하더라도 아래의 최종 안전장치에서 공란을 막습니다.
                pass

            # --- 최종 안전장치: 이유 리스트 공란 방지 (try 밖에서 무조건 실행) ---
            if not positives:
                if p_pd < pd_cut:
                    positives = (neg_auto[:3] if ("neg_auto" in locals() and neg_auto) else [
                        "정상 범위일 가능성이 더 높습니다. (경계 구간이라면 재측정/추가 평가를 권장합니다.)"
                    ])
                else:
                    positives = (neg_auto[:3] if ("neg_auto" in locals() and neg_auto) else [
                        "정상 가능성을 지지하는 근거는 제한적입니다(현재 입력에서는 PD 관련 신호가 더 우세합니다)."
                    ])
            # negatives(=PD 가능성 근거)는 학습데이터 기반/규칙 기반 근거를 합친 뒤에도 비어있을 수 있어,
            # 여기서는 미리 채우지 않고 아래에서 '공란 방지' 로직으로 안전하게 보강합니다.
            # --- Step1 해석 타이틀/순서(확률 구간에 따라) + 설명 공란 방지 ---
            band_code = st.session_state.get("step1_band_code", None)

            # 학습데이터 기반 '가까움' 설명(안전장치)
            training_path = get_training_file()
            train_mtime = None
            if False and training_path and os.path.exists(training_path):
                try:
                    train_mtime = os.path.getmtime(training_path)
                except Exception:
                    train_mtime = None

            stats_step1 = get_step1_training_stats(_file_mtime=train_mtime)
            x_dict = {
                "F0": st.session_state.get("f0_mean", np.nan),
                "Range": st.session_state.get("step1_range_used", range_adj),
                "강도(dB)": final_db,
                "SPS": final_sps,
            }
            n_like, pd_like_strict, pd_like_closest = explain_step1_by_training(stats_step1, x_dict, topk=3)

            # 학습데이터(중앙값) 기반 근거를 보강
            if n_like:
                positives = list(dict.fromkeys((positives or []) + n_like))

            # PD 근거는 '명확히 PD쪽'이 있으면 우선 사용, 없고 cut-off 근처(경계)라면
            # 'PD 중앙값과 상대적으로 가까운 항목'을 보여줘서 공란을 방지합니다.
            borderline = abs(p_pd - pd_cut) <= 0.10
            pd_like = pd_like_strict if pd_like_strict else (pd_like_closest if borderline else [])
            if pd_like:
                negatives = list(dict.fromkeys((negatives or []) + pd_like))

            # 컷오프 근처이면 첫 줄을 경계 안내로 고정(그리고 아래에 "어떤 지표"인지 반드시 보여줌)
            if borderline:
                border_note = f"PD 확률이 cut-off({pd_cut:.2f}) 근처의 **경계 구간**입니다(PD={p_pd*100:.1f}%). 아래 지표를 중심으로 추가 평가/재측정을 권장합니다."
                # 경계인데도 negatives가 비어있으면(입력 누락/통계 없음) 한 줄은 보장
            if not negatives:
                # 기여도 기반 근거가 있으면 그걸 우선 표시(구체적 지표가 포함됨)
                if ("pos_auto" in locals()) and pos_auto:
                    negatives = pos_auto[:3]
                else:
                    dist = abs(p_pd - pd_cut)
                    if dist <= 0.05:
                        negatives = [f"PD 확률이 cut-off({pd_cut:.2f}) 근처입니다(PD={p_pd*100:.1f}%). 재측정/추가 평가로 확인이 필요합니다."]
                    else:
                        negatives = [f"PD 확률이 cut-off({pd_cut:.2f})를 초과했습니다(PD={p_pd*100:.1f}%). 음향 지표 중 일부가 PD 학습군과 겹칠 수 있어 재측정/추가 평가로 확인이 필요합니다."]
            # 타이틀 톤: 더 높은 쪽(주결론) 먼저 보여주기
            primary_is_pd = bool(p_pd >= pd_cut)

            band_suffix = {
                "normal_very_high": "(매우 높음)",
                "normal_high": "(높음)",
                "border_mixed": "(경계)",
                "border_mixed_normal_lean": "(경계, 정상 우세)",
                "border_mixed_normal_lean2": "(경계, 정상 우세)",
                "border_cutoff": "(컷오프 근처)",
                "pd_possible": "(가능성)",
                "pd_high": "(높음)",
                "pd_very_high": "(매우 높음)",
            }.get(band_code, "")

            if primary_is_pd:
                title_primary = f"##### 🔴 파킨슨 가능성을 시사하는 근거 {band_suffix}".strip()
                title_secondary = "##### ✅ 정상 가능성을 지지하는 근거"
                list_primary, list_secondary = negatives, positives
            else:
                title_primary = f"##### ✅ 정상 가능성을 지지하는 근거 {band_suffix}".strip()
                title_secondary = "##### ⚠️ 파킨슨 가능성을 시사하는 근거"
                list_primary, list_secondary = positives, negatives

            st.markdown(title_primary)
            for t in (list_primary or []):
                st.write(f"- {t}")

            st.markdown(title_secondary)
            for t in (list_secondary or []):
                st.write(f"- {t}")

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
            st.error(f"모델 로드 실패: {MODEL_LOAD_ERROR or 'training_data 파일/컬럼/인코딩을 확인하세요.'}")

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
