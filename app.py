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

# --- Step1 학습 통계(가드/해석용) ---
STATS_STEP1 = {}

# --- 내장 training_data (파일이 없을 때 fallback) ---
TRAINING_DATA_CSV_EMBED = r"""
환자ID,성별,F0,Range,강도(dB),SPS,음도(청지각),음도범위(청지각),강도(청지각),말속도(청지각),조음정확도(청지각),VHI총점,VHI_신체,VHI_기능,VHI_정서,진단결과 (Label)
PD1,여,193.6,137.78,56.57,4.56,52.78,48.89,36.33,66.22,59.67,58,20,16,22,PD_Intensity
PD2,남,202.21,148.07,61.03,2.38,56,51.56,67.89,36.56,51.56,58,19,19,20,PD_Rate
PD3,여,174.1,61.33,54.05,4.08,33.44,34.22,10.22,50.11,66.89,27,15,7,5,PD_Intensity
PD4,남,141.27,134.37,61,3.75,46,51.89,61.44,51.78,40.78,67,21,23,23,PD_Articulation
PD5,여,155.74,106.17,51.07,3.23,35.67,35,23,41.78,37.78,56,18,18,20,PD_Intensity
PD6,남,179.69,151.93,67.91,3.97,53,63.33,69.89,55.44,26.44,58,23,19,16,PD_Articulation
PD7,여,126.97,69.55,51.78,3.26,22.78,17,12.78,40.78,20.33,116,36,40,40,PD_Intensity
PD8,여,169.32,105.57,56.26,4.42,46.89,47.78,34.78,50.56,61.11,68,23,22,23,PD_Intensity
PD9,남,114.93,54.89,55.03,4.58,24.56,18.11,19.44,66.44,23.78,37,14,13,10,PD_Rate
PD10,남,122.54,78.4,58.81,3.36,33.89,37.89,31.78,39.56,60.56,36,16,10,10,PD_Intensity
PD11,남,113.83,92.63,59.85,3.93,43.56,33.11,63.22,58.67,45.11,55,23,19,13,PD_Articulation
PD12,남,124.23,88.15,57.35,3.26,43.56,53.56,49.89,47.56,60.56,66,23,24,19,PD_Articulation
PD13,남,138.56,102.52,63.63,7.03,48.22,34.67,60.22,92.33,37.11,96,24,35,37,PD_Rate
PD14,여,198.33,68.22,50.58,3.97,29.44,13,6.78,40.44,24.89,87,27,30,30,PD_Intensity
PD15,남,131.23,91.37,58.75,3.55,52.67,56.33,58.11,68.33,71.89,33,15,11,7,PD_Intensity
PD16,여,189.72,111.82,62.57,2.64,55,35.44,61.44,59,61.56,57,21,18,18,PD_Intensity
PD17,여,165.65,139.99,51.45,3.78,61.56,63.11,46.44,58.67,77.67,30,9,11,10,PD_Articulation
PD18,여,154.43,103.33,52.59,3.59,41.33,35.78,27.67,52.11,49.67,60,20,20,20,PD_Intensity
PD19,남,154.52,112.58,60.23,2.97,36,40,22,55,19,86,26,29,31,PD_Articulation
PD20,여,198.38,120.44,60.32,4.85,52.78,48.89,36.33,66.22,59.67,58,20,16,22,PD_Intensity
PD21,남,162.44,90.93,60.31,2.61,56,51.56,67.89,36.56,51.56,58,19,19,20,PD_Rate
PD22,여,163.2,61.33,58.7,4.6,33.44,34.22,10.22,50.11,66.89,27,15,7,5,PD_Intensity
PD23,남,149.97,127.12,64.11,3.48,46,51.89,61.44,51.78,40.78,67,21,23,23,PD_Articulation
PD24,여,145.68,73.18,57.33,3.26,35.67,35,23,41.78,37.78,56,18,18,20,PD_Intensity
PD25,남,182.61,128.94,70.04,3.62,53,63.33,69.89,55.44,26.44,58,23,19,16,PD_Articulation
PD26,여,130.67,103.77,56.67,4.04,22.78,17,12.78,40.78,20.33,116,36,40,40,PD_Intensity
PD27,여,165.23,108.26,58.28,3.99,46.89,47.78,34.78,50.56,61.11,68,23,22,23,PD_Intensity
PD28,남,109.27,39.44,61.98,6.73,24.56,18.11,19.44,66.44,23.78,37,14,13,10,PD_Rate
PD29,남,119.13,70.8,60.43,3.43,33.89,37.89,31.78,39.56,60.56,36,16,10,10,PD_Intensity
PD30,남,109.61,61.91,63.79,4.14,43.56,33.11,63.22,58.67,45.11,55,23,19,13,PD_Articulation
PD31,남,123.98,90.91,62.98,3.69,43.56,53.56,49.89,47.56,60.56,66,23,24,19,PD_Articulation
PD32,남,131.23,71.17,64.67,7.36,48.22,34.67,60.22,92.33,37.11,96,24,35,37,PD_Rate
PD33,여,192.23,75.1,53.2,3.79,29.44,13,6.78,40.44,24.89,87,27,30,30,PD_Intensity
PD34,남,131.25,86.03,66.02,4.71,52.67,56.33,58.11,68.33,71.89,33,15,11,7,PD_Intensity
PD35,여,176.26,94.82,68.39,3.75,55,35.44,61.44,59,61.56,57,21,18,18,PD_Intensity
PD36,여,180.87,130.98,62.97,4.06,61.56,63.11,46.44,58.67,77.67,30,9,11,10,PD_Articulation
PD37,여,151.7,115.13,60.82,4.3,41.33,35.78,27.67,52.11,49.67,60,20,20,20,PD_Intensity
PD38,남,157,91.94,67.62,3.46,36,40,22,55,19,86,26,29,31,PD_Articulation
CG39,여,171.58,165.25,70.75,3.69,,,,,,0,0,0,0,normal
CG40,여,210.07,130.17,72.84,4.03,,,,,,19,9,8,2,normal
CG41,여,185.79,121.22,71.17,4.07,,,,,,0,0,0,0,normal
CG42,여,212.86,204.65,73.57,4.02,,,,,,3,2,1,0,normal
CG43,남,119.32,122.65,71.66,4.17,,,,,,4,1,3,0,normal
CG44,여,163.08,99.27,70.24,4.13,,,,,,8,3,1,4,normal
CG45,여,199.02,154.42,68.27,3.69,,,,,,3,2,1,0,normal
CG46,여,205.24,147.31,70.24,3.53,,,,,,1,0,1,0,normal
CG47,여,203.18,122.49,64.6,3.54,,,,,,20,8,8,4,normal
CG48,남,91.15,69.69,69.06,3.11,,,,,,0,0,0,0,normal
CG49,여,173.58,167.27,64.58,3.1,,,,,,42,17,11,14,normal
CG50,여,208.25,158.35,72.85,3.96,,,,,,48,18,14,16,normal
CG51,여,163.14,175.14,66.31,3.55,,,,,,2,2,0,0,normal
CG52,여,201.2,159.19,61.21,3.86,,,,,,4,1,2,1,normal
CG53,여,176.88,190.2,70.4,4.56,,,,,,16,4,7,5,normal
CG54,여,188.55,165.02,69.34,4.34,,,,,,5,5,0,0,normal
CG55,여,183.81,175.35,69.69,4.4,,,,,,0,0,0,0,normal
CG56,여,180.75,89.77,67.21,4.16,,,,,,0,0,0,0,normal
CG57,남,163.93,164.48,71.88,3.27,,,,,,0,0,0,0,normal
CG58,여,170.43,188.24,66.12,2.84,,,,,,9,3,4,2,normal
CG59,남,151.7,95.19,68.1,3.71,,,,,,0,0,0,0,normal
CG60,남,137.66,110.73,72.72,3.24,,,,,,3,3,0,0,normal
CG61,남,160.12,128.14,67.32,3.93,,,,,,0,0,0,0,normal
CG62,남,165.97,96.16,77.38,3.74,,,,,,38,16,12,10,normal
CG63,남,147.64,153.77,65.49,3.31,,,,,,0,0,0,0,normal
CG64,여,188.73,188.42,67.28,3.95,,,,,,4,4,0,0,normal
CG65,여,201.86,123.81,68.07,4.64,,,,,,0,0,0,0,normal
CG66,여,185.54,170.03,67.43,3.62,,,,,,0,0,0,0,normal
CG67,여,186.14,141.1,75.65,4.44,,,,,,7,4,1,2,normal
CG68,여,204.5,201.14,72.86,4.84,,,,,,13,5,7,1,normal
CG69,여,208.76,130.92,71.97,4.56,,,,,,0,0,0,0,normal
CG70,여,201.53,175.22,70.19,4.35,,,,,,1,0,1,0,normal
CG71,여,199.68,172.65,70.25,4.6,,,,,,11,4,3,4,normal
CG72,여,211.44,178.91,68.08,3.19,,,,,,10,6,4,0,normal
CG73,여,176.56,155.94,69.94,4.22,,,,,,18,7,10,1,normal
CG74,여,165.93,129.68,70.8,4.1,,,,,,11,9,1,1,normal
CG75,여,196.82,125.85,71.96,3.39,,,,,,32,8,14,10,normal
CG76,여,189.85,155.54,73.97,3.91,,,,,,0,0,0,0,normal
CG77,여,203.51,186.16,76.39,4.53,,,,,,4,1,1,2,normal
CG78,남,120.86,53.35,74.76,4.52,,,,,,0,0,0,0,normal
CG79,여,168.65,119.56,72.79,3.71,,,,,,19,9,8,2,normal
CG80,여,195.34,150.3,72.54,3.92,,,,,,0,0,0,0,normal
CG81,여,197.36,137.15,73.28,3.94,,,,,,3,2,1,0,normal
CG82,여,193.88,98.7,70.33,4,,,,,,4,1,3,0,normal
CG83,남,90.96,104.96,70.64,3.33,,,,,,8,3,1,4,normal
CG84,여,174.02,208.07,67.66,3.25,,,,,,3,2,1,0,normal
CG85,여,204.82,154.52,74.01,3.77,,,,,,1,0,1,0,normal
CG86,여,156.21,128.4,67.43,3.78,,,,,,20,8,8,4,normal
CG87,여,206.71,146.91,72.09,4.09,,,,,,0,0,0,0,normal
CG88,여,179.08,167.22,73,4.94,,,,,,42,17,11,14,normal
CG89,여,181.04,140.7,73.73,4.84,,,,,,48,18,14,16,normal
CG90,여,189.67,96.71,69.72,4.74,,,,,,2,2,0,0,normal
CG91,여,171.85,98.7,67.57,4.55,,,,,,4,1,2,1,normal
CG92,남,157.46,110.24,77.39,3.56,,,,,,16,4,7,5,normal
CG93,여,164.11,156.69,65.95,3.15,,,,,,5,5,0,0,normal
CG94,남,154.23,87.83,72.37,3.94,,,,,,0,0,0,0,normal
CG95,남,145.97,161.93,76.84,2.81,,,,,,0,0,0,0,normal
CG96,남,145.77,149.27,77.06,2.84,,,,,,0,0,0,0,normal
CG97,남,169.1,102.17,77.89,3.41,,,,,,9,3,4,2,normal
CG98,남,156.71,115.34,69.84,3.67,,,,,,0,0,0,0,normal
CG99,여,184.34,146.43,71.85,3.93,,,,,,3,3,0,0,normal
CG100,여,194.48,171.42,70.99,5.24,,,,,,0,0,0,0,normal
CG101,여,182.02,146.87,68.54,3.77,,,,,,38,16,12,10,normal
CG102,여,188.47,161.43,79.46,4.04,,,,,,0,0,0,0,normal
CG103,여,200.9,163.51,75.4,5.01,,,,,,4,4,0,0,normal
CG104,여,210.52,146.35,73.74,4.81,,,,,,0,0,0,0,normal
CG105,여,196.33,148.44,73.27,4.84,,,,,,0,0,0,0,normal
CG106,여,189.94,176.84,71.27,4.64,,,,,,7,4,1,2,normal
CG107,여,209.17,138.84,71.95,4.38,,,,,,13,5,7,1,normal
CG108,여,175.13,155.11,74.85,4.21,,,,,,0,0,0,0,normal
PD109,여,169.96 ,135.41 ,62.90 ,5.06 ,57.32 ,53.53 ,56.67 ,53.53 ,44.16 ,,,,,PD_Articulation
PD110,여,168.89 ,152.24 ,63.27 ,5.48 ,57.32 ,53.53 ,56.67 ,53.53 ,44.16 ,,,,,PD_Articulation
PD111,남,142.43 ,226.98 ,67.98 ,7.14 ,45.95 ,30.95 ,32.95 ,32.21 ,44.05 ,,,,,PD_Intensity
PD112,남,146.16 ,101.26 ,65.43 ,6.02 ,45.95 ,30.95 ,32.95 ,32.21 ,44.05 ,,,,,PD_Intensity
PD113,여,175.23 ,77.24 ,66.63 ,6.26 ,71.53 ,54.79 ,53.58 ,55.58 ,69.32 ,,,,,PD_Intensity
PD114,여,175.50 ,120.20 ,67.79 ,6.41 ,71.53 ,54.79 ,53.58 ,55.58 ,69.32 ,,,,,PD_Intensity
PD115,여,204.71 ,83.37 ,64.48 ,5.77 ,59.26 ,42.79 ,36.32 ,36.37 ,57.79 ,,,,,PD_Intensity
PD116,여,190.87 ,179.51 ,65.61 ,6.09 ,59.26 ,42.79 ,36.32 ,36.37 ,57.79 ,,,,,PD_Intensity
PD117,여,190.82 ,143.23 ,64.53 ,5.39 ,49.32 ,59.68 ,44.79 ,44.00 ,34.89 ,,,,,PD_Articulation
PD118,여,188.24 ,140.01 ,65.92 ,6.50 ,49.32 ,59.68 ,44.79 ,44.00 ,34.89 ,,,,,PD_Articulation
PD119,여,176.89 ,90.17 ,66.16 ,6.13 ,56.57 ,44.86 ,39.05 ,42.38 ,59.00 ,,,,,PD_Intensity
PD120,여,145.08 ,159.49 ,60.63 ,6.10 ,56.57 ,44.86 ,39.05 ,42.38 ,59.00 ,,,,,PD_Intensity
PD121,여,210.02 ,161.37 ,67.76 ,6.74 ,73.53 ,63.74 ,49.79 ,46.84 ,71.68 ,,,,,PD_Intensity
PD122,여,196.67 ,137.42 ,68.11 ,7.41 ,73.53 ,63.74 ,49.79 ,46.84 ,71.68 ,,,,,PD_Intensity
PD123,남,134.28 ,95.15 ,58.43 ,6.52 ,17.79 ,19.63 ,23.84 ,27.89 ,33.95 ,,,,,PD_Intensity
PD124,남,138.02 ,122.81 ,66.99 ,7.44 ,17.79 ,19.63 ,23.84 ,27.89 ,33.95 ,,,,,PD_Intensity
PD125,남,149.04 ,71.62 ,73.14 ,5.12 ,27.89 ,29.68 ,34.00 ,35.11 ,27.37 ,,,,,PD_Articulation
PD126,남,152.58 ,143.66 ,68.24 ,5.74 ,27.89 ,29.68 ,34.00 ,35.11 ,27.37 ,,,,,PD_Articulation
PD127,여,204.92 ,137.26 ,71.94 ,6.68 ,68.42 ,57.63 ,46.89 ,55.68 ,73.74 ,,,,,PD_Intensity
PD128,여,197.40 ,129.48 ,68.14 ,6.60 ,68.42 ,57.63 ,46.89 ,55.68 ,73.74 ,,,,,PD_Intensity
PD129,남,84.70 ,50.42 ,60.63 ,7.37 ,35.50 ,36.00 ,31.65 ,58.15 ,47.25 ,,,,,PD_Intensity
PD130,남,93.21 ,98.55 ,60.95 ,7.69 ,35.50 ,36.00 ,31.65 ,58.15 ,47.25 ,,,,,PD_Intensity
PD131,남,116.92 ,35.87 ,63.20 ,8.91 ,61.15 ,60.30 ,52.50 ,44.70 ,75.30 ,,,,,PD_Intensity
PD132,남,120.86 ,33.92 ,71.56 ,7.91 ,61.15 ,60.30 ,52.50 ,44.70 ,75.30 ,,,,,PD_Intensity
PD133,남,98.25 ,47.43 ,64.76 ,8.04 ,24.50 ,31.75 ,29.35 ,56.20 ,61.45 ,,,,,PD_Intensity
PD134,남,95.59 ,131.21 ,64.84 ,8.56 ,24.50 ,31.75 ,29.35 ,56.20 ,61.45 ,,,,,PD_Intensity
PD135,남,144.64 ,64.81 ,61.22 ,8.58 ,19.50 ,25.27 ,18.68 ,46.45 ,35.73 ,,,,,PD_Intensity
PD136,남,146.31 ,75.63 ,62.34 ,9.82 ,19.50 ,25.27 ,18.68 ,46.45 ,35.73 ,,,,,PD_Intensity
PD137,여,191.70 ,108.76 ,64.84 ,5.29 ,7.95 ,8.40 ,10.05 ,34.80 ,7.00 ,,,,,PD_Articulation
PD138,여,199.23 ,176.08 ,66.23 ,5.81 ,7.95 ,8.40 ,10.05 ,34.80 ,7.00 ,,,,,PD_Articulation
"""

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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# -------------------------------
# Step1 screening 메시지(확률 구간별)
# -------------------------------
def step1_screening_band(p_pd: float, pd_cut: float = 0.50):
    try:
        p_pd = float(p_pd)
    except Exception:
        p_pd = 0.0
    if not np.isfinite(p_pd):
        p_pd = 0.0
    p_pd = max(0.0, min(1.0, p_pd))
    p_norm = 1.0 - p_pd

    if p_norm >= 0.8:
        if p_norm >= 0.9:
            return ("success", "정상 가능성이 매우 높습니다.", "normal_very_high")
        return ("success", "정상에 매우 가깝습니다.", "normal_high")
    if p_norm >= 0.6:
        return ("success", "정상 가능성이 높습니다.", "normal_mid")
    if p_norm >= 0.4:
        return ("warning", f"경계 구간입니다(컷오프 {pd_cut:.2f} 기준). 재측정/추가 평가를 권장합니다.", "border")
    if p_norm >= 0.2:
        return ("warning", "파킨슨 가능성이 있습니다(임상 소견과 함께 해석).", "pd_possible")
    return ("error", "파킨슨 가능성이 높습니다(추가 평가 권장).", "pd_high")

@st.cache_data
def get_step1_training_stats(_file_mtime=None):
    training_path = get_training_file()
    if training_path is None:
        return None

    try:
        df = pd.read_csv(training_path) if str(training_path).lower().endswith(".csv") else pd.read_excel(training_path)
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
    if not stats:
        return [], [], []

    reasons_pd_strict, reasons_n, reasons_pd_closest = [], [], []
    scored = []

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

        strength = float((d_n - d_pd) / denom)
        abs_strength = abs(strength)
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

    for _, strength, closeness_pd, f, x, pd_med, n_med in scored:
        name, fmt, pd_fmt, n_fmt = _fmt(f, x, pd_med, n_med)

        if strength > 0 and len(reasons_pd_strict) < topk:
            reasons_pd_strict.append(f"{name}가 {fmt}로 **정상 중앙값({n_fmt})보다 PD 중앙값({pd_fmt})에 더 가깝습니다**.")
        elif strength < 0 and len(reasons_n) < topk:
            reasons_n.append(f"{name}가 {fmt}로 **PD 중앙값({pd_fmt})보다 정상 중앙값({n_fmt})에 더 가깝습니다**.")

        if len(reasons_pd_strict) >= topk and len(reasons_n) >= topk:
            break

    scored_by_pd = sorted(scored, reverse=True, key=lambda t: t[2])
    for _, strength, closeness_pd, f, x, pd_med, n_med in scored_by_pd:
        if len(reasons_pd_closest) >= topk:
            break
        name, fmt, pd_fmt, n_fmt = _fmt(f, x, pd_med, n_med)
        reasons_pd_closest.append(
            f"{name}가 {fmt}이며, **PD 중앙값({pd_fmt})과의 거리가 비교적 가깝습니다**(정상 중앙값 {n_fmt})."
        )

    return reasons_n[:topk], reasons_pd_strict[:topk], reasons_pd_closest[:topk]

def _safe_float(x, default=None):
    try:
        if x is None:
            return default
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

FEAT_LABELS_STEP1 = {
    "F0_Z": "평균 음도(F0, 성별 표준화 z)",
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
    imputer = None
    scaler = None
    est = pipeline
    try:
        if hasattr(pipeline, "named_steps"):
            steps = pipeline.named_steps
            for key in ("clf", "logit", "lr", "lda", "qda", "model", "imp", "sc"):
                if key in steps:
                    pass
            est = list(steps.values())[-1]
            imputer = steps.get("imputer") or steps.get("imp")
            scaler = steps.get("scaler") or steps.get("sc")
    except Exception:
        pass
    return imputer, scaler, est

def top_contrib_linear_multiclass(pipeline, x_row, feat_names, pred_class, topk=3):
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
        cidx = 0

    w = coef[cidx] if getattr(coef, "ndim", 1) == 2 else coef
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
except Exception:
    st.warning("⚠️ Secrets 설정이 없어 구글시트/이메일 전송은 비활성화됩니다. (SQLite 저장은 사용 가능)")
    SHEET_NAME = None
    HAS_GCP_SECRETS = False

# ==========================================
# [전역 설정] 폰트 및 변수
# ==========================================
FEATS_STEP1 = ["F0_Z", "Intensity", "SPS"]
FEATS_STEP2 = ['Intensity', 'SPS', 'P_Loudness', 'P_Rate', 'P_Artic']
MIX_MARGIN_P = 0.10

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

F0Z_STATS = None

def _compute_f0z_stats(df: pd.DataFrame):
    def _safe_mu_sd(arr):
        arr = pd.to_numeric(pd.Series(arr), errors="coerce").dropna().astype(float).values
        if arr.size < 3:
            return None
        mu = float(np.nanmean(arr))
        sd = float(np.nanstd(arr, ddof=0))
        if not np.isfinite(sd) or sd <= 1e-6:
            sd = 1.0
        return (mu, sd)

    f0_all = pd.to_numeric(df.get("F0", np.nan), errors="coerce")
    sex_all = df.get("성별", None)
    diag_raw = df.get("진단결과 (Label)", None)

    normal_mask = pd.Series([False]*len(df))
    if diag_raw is not None:
        tmp = diag_raw.astype(str).str.lower()
        normal_mask = tmp.str.contains("normal")

    def _f0_by_sex(mask):
        male = []
        female = []
        for i in range(len(df)):
            if not bool(mask.iloc[i]):
                continue
            sx = sex_to_num(sex_all.iloc[i] if hasattr(sex_all, "iloc") else None)
            f0 = f0_all.iloc[i] if hasattr(f0_all, "iloc") else None
            if f0 is None or (not np.isfinite(f0)):
                continue
            if sx >= 0.75:
                male.append(f0)
            elif sx <= 0.25:
                female.append(f0)
        return male, female

    male_n, female_n = _f0_by_sex(normal_mask)
    male_a, female_a = _f0_by_sex(pd.Series([True]*len(df)))

    male_stats = _safe_mu_sd(male_n) or _safe_mu_sd(male_a) or (120.0, 25.0)
    female_stats = _safe_mu_sd(female_n) or _safe_mu_sd(female_a) or (210.0, 35.0)
    global_stats = _safe_mu_sd(f0_all) or (170.0, 50.0)
    return {"male": male_stats, "female": female_stats, "global": global_stats}

def _f0_to_z(f0_value, sex_num):
    try:
        f0 = float(f0_value)
    except Exception:
        return np.nan
    try:
        sx = sex_to_num(sex_num)
        if isinstance(F0Z_STATS, dict):
            if sx >= 0.75:
                mu, sd = F0Z_STATS.get("male", (120.0, 25.0))
            elif sx <= 0.25:
                mu, sd = F0Z_STATS.get("female", (210.0, 35.0))
            else:
                mu, sd = F0Z_STATS.get("global", (170.0, 50.0))
        else:
            if sx >= 0.75:
                mu, sd = (120.0, 25.0)
            elif sx <= 0.25:
                mu, sd = (210.0, 30.0)
            else:
                mu, sd = (165.0, 35.0)
        sd = float(sd) if (sd is not None and np.isfinite(sd) and sd > 1e-6) else 1.0
        return (f0 - float(mu)) / sd
    except Exception:
        return np.nan

MODEL_LOAD_ERROR = ""

def get_training_file():
    base = Path(__file__).resolve().parent
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
    for p in base.rglob("training_data.csv"):
        return p
    for p in base.rglob("training_data.xlsx"):
        return p
    return None

@st.cache_resource
def _youden_cutoff(y_true, scores):
    fpr, tpr, thr = roc_curve(y_true, scores)
    j = tpr - fpr
    bi = int(np.argmax(j))
    cut = float(thr[bi]) if np.isfinite(thr[bi]) else 0.5
    sens = float(tpr[bi])
    spec = float(1.0 - fpr[bi])
    return cut, sens, spec

@st.cache_data
def compute_cutoffs_from_training(_file_mtime=None):
    training_path = get_training_file()
    if training_path is None:
        global MODEL_LOAD_ERROR
        MODEL_LOAD_ERROR = "training_data.csv/xlsx 파일을 찾지 못했습니다. app.py와 같은 폴더에 training_data.csv/xlsx를 두세요."
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

    # local f0z stats (independent)
    local_stats = _compute_f0z_stats(df_raw)
    def f0_to_z_local(f0, sex):
        try:
            f0 = float(f0)
        except Exception:
            return np.nan
        sx = sex_to_num(sex)
        if sx >= 0.75:
            mu, sd = local_stats.get("male", (120.0, 25.0))
        elif sx <= 0.25:
            mu, sd = local_stats.get("female", (210.0, 35.0))
        else:
            mu, sd = local_stats.get("global", (170.0, 50.0))
        sd = float(sd) if (sd is not None and np.isfinite(sd) and sd > 1e-6) else 1.0
        return (f0 - float(mu)) / sd

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
        elif 'pd_articulation' in l or 'pd_artic' in l:
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

    df = pd.DataFrame(data_list, columns=['F0','Range','Intensity','SPS','VHI_Total','VHI_P','VHI_F','VHI_E','Sex','P_Pitch','P_Range','P_Loudness','P_Rate','P_Artic','Diagnosis','Subgroup'])

    # numeric
    for col in ['F0','Range','Intensity','SPS','Sex','P_Pitch','P_Range','P_Loudness','P_Rate','P_Artic','VHI_Total','VHI_P','VHI_F','VHI_E']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # fill
    for col in ['F0','Range','Intensity','SPS','P_Pitch','P_Range','P_Loudness','P_Rate','P_Artic','Sex']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())
    for col in ['VHI_Total','VHI_P','VHI_F','VHI_E']:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # F0_Z
    df['F0_Z'] = [f0_to_z_local(df.loc[i,'F0'], df.loc[i,'Sex']) for i in df.index]

    # Step1
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
    step1_cutoff_auto, step1_sens, step1_spec = _youden_cutoff(y1_bin, oof_pd)
    step1_cutoff = 0.50  # 운영 고정

    # Step2
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
            y_tr = y2[tr]
            if len(np.unique(y_tr)) < 2:
                continue
            pipe2.fit(X2.iloc[tr], y_tr)
            proba = pipe2.predict_proba(X2.iloc[te])[0]
            fold_classes = pipe2.named_steps["clf"].classes_
            for j, c in enumerate(fold_classes):
                oof2[te[0], class_to_idx[c]] = float(proba[j])

        for c in classes:
            y_bin = (y2 == c).astype(int)
            p = oof2[:, class_to_idx[c]]
            if np.all(y_bin == 0) or np.all(y_bin == 1):
                cutoff_by_class[c] = 0.5
                continue
            cut, _, _ = _youden_cutoff(y_bin, p)
            cutoff_by_class[c] = float(cut)

        y_pred = [classes[int(np.argmax(oof2[i]))] for i in range(len(df_pd))]
        step2_cm = confusion_matrix(y2, y_pred, labels=list(classes))
        step2_report = {"classes": list(classes), "confusion_matrix": step2_cm.tolist()}

    y_pred1 = (oof_pd >= step1_cutoff).astype(int)
    step1_cm = confusion_matrix(y1_bin, y_pred1, labels=[0, 1])

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
        except Exception:
            pass
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
def train_models(cache_buster: str = "v28_7_0"):
    global MODEL_LOAD_ERROR, F0Z_STATS, STATS_STEP1

    training_path = get_training_file()

    try:
        if training_path is not None and str(training_path).lower().endswith(".xlsx"):
            raw = pd.read_excel(training_path)
        elif training_path is not None:
            raw = pd.read_csv(training_path, encoding="utf-8-sig")
        else:
            raw = pd.read_csv(io.StringIO(TRAINING_DATA_CSV_EMBED), encoding='utf-8-sig')
    except Exception as e:
        MODEL_LOAD_ERROR = f"training_data 로드 실패: {type(e).__name__}: {e}"
        return None, None

    if '진단결과 (Label)' not in raw.columns:
        MODEL_LOAD_ERROR = "training_data에 '진단결과 (Label)' 컬럼이 없습니다."
        return None, None

    try:
        F0Z_STATS = _compute_f0z_stats(raw)
    except Exception:
        F0Z_STATS = {"male": (120.0, 25.0), "female": (210.0, 35.0), "global": (170.0, 50.0)}

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
    sex_list = []
    range_list = []
    f0_raw_list = []

    for _, row in raw.iterrows():
        diag, _sub = _label_to_diag_and_sub(row.get('진단결과 (Label)', ''))
        if diag is None:
            continue

        sex_num = sex_to_num(row.get('성별', None))
        f0_raw = row.get('F0', np.nan)
        range_val = row.get('Range', np.nan)
        f0_z = _f0_to_z(f0_raw, sex_num)

        X1_rows.append([f0_z, row.get('강도(dB)', np.nan), row.get('SPS', np.nan)])
        sex_list.append(sex_num)
        range_list.append(range_val)
        f0_raw_list.append(f0_raw)
        y1.append(diag)

    X1 = np.array(X1_rows, dtype=float)

    # Step1 학습 데이터 기반 기준값(가드/해석용)
    try:
        X1_arr = np.array(X1_rows, dtype=float)
        y1_arr = np.array(y1, dtype=str)
        sex_arr = np.array(sex_list, dtype=float)
        range_arr = np.array(range_list, dtype=float)
        f0_raw_arr = np.array([pd.to_numeric(x, errors="coerce") for x in f0_raw_list], dtype=float)

        _db_all = X1_arr[:, 1] if X1_arr.ndim == 2 and X1_arr.shape[1] >= 2 else np.array([])
        _sps_all = X1_arr[:, 2] if X1_arr.ndim == 2 and X1_arr.shape[1] >= 3 else np.array([])
        _is_norm = (y1_arr == "Normal")

        median_range_all = float(np.nanmedian(range_arr)) if np.any(np.isfinite(range_arr)) else 100.0
        if np.any((sex_arr == 1) & np.isfinite(range_arr)):
            median_range_m = float(np.nanmedian(range_arr[(sex_arr == 1)]))
        else:
            median_range_m = median_range_all
        if np.any((sex_arr == 0) & np.isfinite(range_arr)):
            median_range_f = float(np.nanmedian(range_arr[(sex_arr == 0)]))
        else:
            median_range_f = median_range_all

        if np.any(_is_norm) and _db_all.size and _sps_all.size:
            _db_norm = _db_all[_is_norm]
            _sps_norm = _sps_all[_is_norm]
            db_p05, db_p50, db_p95 = [float(np.nanpercentile(_db_norm, q)) for q in (5, 50, 95)]
            sps_p95 = float(np.nanpercentile(_sps_norm, 95))
        else:
            db_p05, db_p50, db_p95 = 65.0, 70.0, 76.0
            sps_p95 = 5.0

        if np.any(_is_norm):
            f0_norm = f0_raw_arr[_is_norm]
            sex_norm = sex_arr[_is_norm]
            f0_mu_all = float(np.nanmean(f0_norm)) if np.any(np.isfinite(f0_norm)) else 165.0
            f0_sd_all = float(np.nanstd(f0_norm, ddof=1)) if np.sum(np.isfinite(f0_norm)) >= 2 else 35.0

            f0_norm_m = f0_norm[sex_norm == 1]
            f0_norm_f = f0_norm[sex_norm == 0]
            f0_mu_m = float(np.nanmean(f0_norm_m)) if np.any(np.isfinite(f0_norm_m)) else 120.0
            f0_sd_m = float(np.nanstd(f0_norm_m, ddof=1)) if np.sum(np.isfinite(f0_norm_m)) >= 2 else 25.0
            f0_mu_f = float(np.nanmean(f0_norm_f)) if np.any(np.isfinite(f0_norm_f)) else 210.0
            f0_sd_f = float(np.nanstd(f0_norm_f, ddof=1)) if np.sum(np.isfinite(f0_norm_f)) >= 2 else 30.0
        else:
            f0_mu_all, f0_sd_all = 165.0, 35.0
            f0_mu_m, f0_sd_m = 120.0, 25.0
            f0_mu_f, f0_sd_f = 210.0, 30.0

        STATS_STEP1 = {
            "median_range_all": median_range_all,
            "median_range_m": median_range_m,
            "median_range_f": median_range_f,
            "db_p05_norm": db_p05,
            "db_p50_norm": db_p50,
            "db_p95_norm": db_p95,
            "sps_p95_norm": sps_p95,
            "f0_mu_all": f0_mu_all,
            "f0_sd_all": f0_sd_all,
            "f0_mu_m": f0_mu_m,
            "f0_sd_m": f0_sd_m,
            "f0_mu_f": f0_mu_f,
            "f0_sd_f": f0_sd_f,
        }
    except Exception:
        STATS_STEP1 = {
            "median_range_all": 100.0,
            "median_range_m": 90.0,
            "median_range_f": 120.0,
            "db_p05_norm": 65.0,
            "db_p50_norm": 70.0,
            "db_p95_norm": 76.0,
            "sps_p95_norm": 5.0,
            "f0_mu_all": 165.0,
            "f0_sd_all": 35.0,
            "f0_mu_m": 120.0,
            "f0_sd_m": 25.0,
            "f0_mu_f": 210.0,
            "f0_sd_f": 30.0,
        }

    y1 = np.array(y1, dtype=str)

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
            row.get('강도(dB)', np.nan),
            row.get('SPS', np.nan),
            row.get('강도(청지각)', np.nan),
            row.get('말속도(청지각)', np.nan),
            row.get('조음정확도(청지각)', np.nan)
        ])
        y2.append(sub)

    X2 = np.array(X2_rows, dtype=float)
    y2 = np.array(y2, dtype=object)

    model_step2 = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto"))
    ])
    model_step2.fit(X2, y2)

    return model_step1, model_step2

try:
    model_step1, model_step2 = train_models("v28_7_0")
except Exception as e:
    MODEL_LOAD_ERROR = f"모델 학습 중 예외: {type(e).__name__}: {e}"
    model_step1, model_step2 = None, None

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
        part.add_header("Content-Disposition", f"attachment; filename={email_attach_name}")
        msg.attach(part)

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())
        server.quit()

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
    except Exception:
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
                rng = float(max(0.0, np.max(clean_p) - np.min(clean_p)))
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
    except Exception:
        return None, 0, 0, 0

def run_analysis_logic(file_path, gender=None):
    try:
        g = (gender or "").strip().upper()
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
        st.error(f"분석 오류: {e}")
        return False

def generate_interpretation(prob_normal, db, sps, range_val, artic, vhi, vhi_e, sex=None):
    positives, negatives = [], []
    if vhi < 15:
        positives.append(f"환자 본인의 주관적 불편함(VHI {vhi}점)이 낮아, 일상 대화에 심리적/기능적 부담이 적은 상태입니다.")
    try:
        _sex = (sex or "").strip().upper()
    except Exception:
        _sex = ""
    range_thr = 80 if _sex == "M" else 100
    if range_val >= range_thr:
        positives.append(f"음도 범위가 {range_val:.1f}Hz로 넓게 나타나, 목소리에 생동감이 있고 억양의 변화가 자연스럽습니다.")
    else:
        negatives.append(f"음도 범위가 {range_val:.1f}Hz로 좁게 측정되었습니다. 억양 변화가 단조롭게 나타나는 패턴은 PD 학습군과 겹칠 수 있으나, 과제(짧은 문장/단음절)나 Pitch 추정 불안정에서도 발생할 수 있어 재측정이 권장됩니다.")
    if artic >= 75:
        positives.append(f"청지각적 조음 정확도가 {artic}점으로 양호하여, 상대방이 말을 알아듣기에 명료한 상태입니다.")
    if sps < 5.8:
        positives.append(f"말속도가 {sps:.2f} SPS로 측정되었습니다. 말속도는 안정적인 범위입니다.")
    if db >= 60:
        positives.append(f"평균 음성 강도가 {db:.1f} dB로, 일반적인 대화 수준(60dB 이상)의 성량을 튼튼하게 유지하고 있습니다.")

    if db < 60:
        negatives.append(f"평균 음성 강도가 {db:.1f} dB로 낮게 측정되었습니다(※ 마이크/거리/환경에 따라 절대값은 달라질 수 있으며, 본 도구의 모델 기준으로 낮은 편입니다). 이는 파킨슨병에서 흔한 강도 감소(Hypophonia) 패턴과 유사하여 발성 훈련이 필요할 수 있습니다.")
    if sps >= 5.8:
        negatives.append(f"말속도가 {sps:.2f} SPS로 빠른 편입니다. 정상 성인에서도 빠른 말속도는 나타날 수 있으나, 일부 PD 학습군의 말속도/리듬 특징과 겹칠 수 있어 추가 확인이 필요합니다.")
    if artic < 70:
        negatives.append(f"청지각적 조음 정확도가 {artic}점으로 다소 낮습니다. 발음이 불분명해지는 조음 장애(Dysarthria) 징후가 관찰됩니다.")
    if vhi >= 20:
        negatives.append(f"VHI 총점이 {vhi}점으로 높습니다. 환자 스스로 음성 문제로 인한 생활의 불편함과 심리적 위축을 크게 느끼고 있습니다.")
    if vhi_e >= 5:
        negatives.append("특히 VHI 정서(E) 점수가 높아, 말하기에 대한 불안감이나 자신감 저하가 감지됩니다.")
    return positives, negatives

# --- UI Title ---
st.title("파킨슨병 환자 하위유형 분류 프로그램")
st.markdown("이 프로그램은 청지각적 평가, 음향학적 분석, 자가보고(VHI-10) 데이터를 통합하여 파킨슨병 환자의 음성 특성을 3가지 하위 유형으로 분류합니다.")

# 1. 사이드바
with st.sidebar:
    st.header("👤 대상자 정보 (필수)")
    subject_name = st.text_input("이름 (실명/ID)", "참여자")
    subject_age = st.number_input("나이", 1, 120, 60)
    subject_gender = st.selectbox("성별", ["남", "여"])

# 2. 데이터 수집
st.header("1. 음성 데이터 수집")
if 'user_syllables' not in st.session_state:
    st.session_state.user_syllables = 80
if 'source_type' not in st.session_state:
    st.session_state.source_type = None

col_rec, col_up = st.columns(2)
TEMP_FILENAME = "temp_for_analysis.wav"

with col_rec:
    st.markdown("#### 🎙️ 마이크 녹음")
    font_size = st.slider("🔍 글자 크기", 15, 50, 28, key="fs_read")

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

    syllables_rec = st.number_input("전체 음절 수", 1, 500, default_syl, key=f"syl_rec_{read_opt}")
    st.session_state.user_syllables = syllables_rec

    audio_buf = st.audio_input("낭독 녹음")
    if st.button("🎙️ 녹음된 음성 분석"):
        if audio_buf:
            with open(TEMP_FILENAME, "wb") as f:
                f.write(audio_buf.read())
            st.session_state.current_wav_path = os.path.join(os.getcwd(), TEMP_FILENAME)
            gcode = "M" if subject_gender == "남" else "F"
            run_analysis_logic(st.session_state.current_wav_path, gcode)
        else:
            st.warning("녹음부터 해주세요.")

with col_up:
    st.markdown("#### 📂 파일 업로드")
    up_file = st.file_uploader("WAV 파일 선택", type=["wav"])
    if up_file:
        st.audio(up_file, format='audio/wav')
    if st.button("📂 업로드 파일 분석"):
        if up_file:
            with open(TEMP_FILENAME, "wb") as f:
                f.write(up_file.read())
            st.session_state.current_wav_path = os.path.join(os.getcwd(), TEMP_FILENAME)
            gcode = "M" if subject_gender == "남" else "F"
            run_analysis_logic(st.session_state.current_wav_path, gcode)
        else:
            st.warning("파일을 올려주세요.")

# 3. 결과 및 저장
if st.session_state.get('is_analyzed'):
    st.markdown("---")
    st.subheader("2. 분석 결과 및 보정")

    c1, c2 = st.columns([2, 1])

    with c1:
        st.plotly_chart(st.session_state['fig_plotly'], use_container_width=True)

    with c2:
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

        final_db = float(st.session_state['mean_db']) + float(db_adj)
        range_adj = st.slider("음도범위(Hz) 보정", 0.0, 300.0, float(st.session_state['pitch_range']))

        s_time, e_time = st.slider(
            "말속도 구간(초)", 0.0, float(st.session_state['duration']),
            st.session_state.get('sps_window', (0.0, float(st.session_state['duration']))),
            0.01, key="sps_window_slider"
        )
        st.session_state['sps_window'] = (float(s_time), float(e_time))
        st.caption("※ 말속도 구간을 바꾸면 SPS(표)는 즉시 바뀝니다. 최종 확률/문구는 [🚀 진단 결과 확인]을 다시 눌러 갱신하세요.")
        sel_dur = max(0.1, float(e_time) - float(s_time))
        final_sps = float(st.session_state.user_syllables) / sel_dur
        st.session_state['sps_final'] = float(final_sps)

        st.write("#### 📊 음향학적 분석 결과")
        result_df = pd.DataFrame({
            "항목": ["평균 강도(dB)", "평균 음도(Hz)", "음도 범위(Hz)", "말속도(SPS)"],
            "수치": [f"{final_db:.2f}", f"{float(st.session_state['f0_mean']):.2f}", f"{float(range_adj):.2f}", f"{float(st.session_state.get('sps_final', final_sps)):.2f}"]
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

        # Session에 저장(진단 시 참조)
        st.session_state["vhi_total"] = float(vhi_total)
        st.session_state["vhi_f"] = float(vhi_f)
        st.session_state["vhi_p"] = float(vhi_p)
        st.session_state["vhi_e"] = float(vhi_e)

        st.markdown("##### 📊 영역별 점수")
        col_v1, col_v2, col_v3, col_v4 = st.columns(4)
        col_v1.metric("총점", f"{vhi_total}점")
        col_v2.metric("기능(F)", f"{vhi_f}점")
        col_v3.metric("신체(P)", f"{vhi_p}점")
        col_v4.metric("정서(E)", f"{vhi_e}점")

    st.markdown("---")
    st.subheader("4. 최종 진단 및 클라우드 전송")

    if st.button("🚀 진단 결과 확인", key="btn_diag"):
        if not model_step1:
            st.error("Step1 모델이 준비되지 않았습니다. training_data 로드/학습을 확인하세요.")
        else:
            pd_cut = 0.50
            p_pd = 0.0
            p_norm = 1.0

            f0_in = _safe_float(st.session_state.get('f0_mean'))
            pr_in = _safe_float(locals().get('range_adj', st.session_state.get('pitch_range')))
            db_in = _safe_float(final_db)
            sps_in = _safe_float(st.session_state.get('sps_final', final_sps))

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

            pr_used = float(pr_in) if (pr_in is not None and np.isfinite(pr_in)) else None
            pr_raw = pr_used

            normal_context = (vhi_now <= 3.0) and (artic_now >= 70.0) and (db_in is not None and db_in >= 65.0) and (sps_in is not None and sps_in <= 5.8)

            db_used = db_in
            sps_used = sps_in
            clamp_msgs = []
            if isinstance(STATS_STEP1, dict):
                db_p05 = float(STATS_STEP1.get("db_p05_norm", 65.0))
                sps_p95 = float(STATS_STEP1.get("sps_p95_norm", 5.0))
                if normal_context and (db_in is not None) and (db_in < db_p05):
                    db_used = db_p05
                    clamp_msgs.append(f"평균 음성 강도(dB)가 정상 학습분포 하한(5퍼센타일≈{db_p05:.1f}dB)보다 낮아, 장비/환경 영향 가능성이 있어 모델 입력은 {db_used:.1f}dB로 보정했습니다. (오탐 방지)")
                if normal_context and (sps_in is not None) and (sps_in > sps_p95) and (sps_in <= 5.6):
                    sps_used = sps_p95
                    clamp_msgs.append(f"말속도(SPS)가 정상 학습분포 상한(95퍼센타일≈{sps_p95:.2f})을 약간 초과해, 모델 입력은 {sps_used:.2f}로 완만히 보정했습니다. (오탐 방지)")
            for _m in clamp_msgs:
                st.info(_m)

            if normal_context and pr_used is not None:
                med_all = float(STATS_STEP1.get('median_range_all', 100.0))
                med_m = float(STATS_STEP1.get('median_range_m', med_all))
                med_f = float(STATS_STEP1.get('median_range_f', med_all))
                med = med_m if sex_is_m else (med_f if sex_is_f else med_all)

                if sex_is_m and pr_used < 70.0:
                    pr_used = med
                elif sex_is_f and pr_used < 90.0:
                    pr_used = med
                elif (not sex_is_m and not sex_is_f) and pr_used < 80.0:
                    pr_used = med

            try:
                st.session_state['step1_range_raw'] = pr_raw
                st.session_state['step1_range_used'] = pr_used
                st.session_state['step1_range_guard'] = bool(normal_context and pr_raw is not None and pr_used != pr_raw)
            except Exception:
                pass

            sex_num_ui = sex_to_num(subject_gender)
            f0_z_in = _f0_to_z(f0_in, sex_num_ui)
            input_1 = pd.DataFrame([[f0_z_in, db_used, sps_used]], columns=FEATS_STEP1)

            proba_1 = model_step1.predict_proba(input_1.to_numpy())[0]
            classes_1 = list(model_step1.classes_)
            if "Parkinson" in classes_1:
                p_pd = float(proba_1[classes_1.index("Parkinson")])
            if "Normal" in classes_1:
                p_norm = float(proba_1[classes_1.index("Normal")])
            else:
                p_norm = 1.0 - p_pd

            prob_normal = p_norm * 100.0

            try:
                if (subject_gender == "남") and (range_adj < 90) and (p_pd < (pd_cut + 0.07)):
                    p_pd = max(0.0, p_pd - 0.07)
                    prob_normal = (1.0 - p_pd) * 100.0
            except Exception:
                pass

            # Step1 판정
            if p_pd >= pd_cut:
                kind, headline, band_code = step1_screening_band(p_pd, pd_cut)
                if kind == "error":
                    st.error(f"🔴 **{headline}**  | Normal={prob_normal:.1f}%  PD={p_pd*100:.1f}% (cut-off={pd_cut:.2f})")
                elif kind == "warning":
                    st.warning(f"🟡 **{headline}**  | Normal={prob_normal:.1f}%  PD={p_pd*100:.1f}% (cut-off={pd_cut:.2f})")
                else:
                    st.success(f"🟢 **{headline}**  | Normal={prob_normal:.1f}%  PD={p_pd*100:.1f}% (cut-off={pd_cut:.2f})")
                final_decision = "Parkinson"
            else:
                kind, headline, band_code = step1_screening_band(p_pd, pd_cut)
                if kind == "warning":
                    st.warning(f"🟡 **{headline}**  | Normal={prob_normal:.1f}%  PD={p_pd*100:.1f}% (cut-off={pd_cut:.2f})")
                elif kind == "error":
                    st.error(f"🔴 **{headline}**  | Normal={prob_normal:.1f}%  PD={p_pd*100:.1f}% (cut-off={pd_cut:.2f})")
                else:
                    st.success(f"🟢 **{headline}**  | Normal={prob_normal:.1f}%  PD={p_pd*100:.1f}% (cut-off={pd_cut:.2f})")
                final_decision = "Normal"

            st.session_state.step1_band_code = band_code

            # Step2 (PD로 판정된 경우)
            if (final_decision == "Parkinson") and model_step2:
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

                # 혼합형 판단
                _pairs = sorted(zip(sub_classes, probs_sub), key=lambda x: float(x[1]), reverse=True)
                _top1_lbl, _top1_p = _pairs[0][0], float(_pairs[0][1])
                _top2_lbl, _top2_p = (_pairs[1][0], float(_pairs[1][1])) if len(_pairs) > 1 else (None, 0.0)
                _is_mixed = (_top2_lbl is not None) and ((_top1_p - _top2_p) < MIX_MARGIN_P)

                if _is_mixed and _top2_lbl is not None:
                    st.info(f"➡️ PD 하위 집단 예측 : 혼합형으로 {_top1_lbl} 가능성이 더 높고, {_top2_lbl} 문제를 동반할 수 있습니다({_top1_lbl} {_top1_p*100:.1f}%, {_top2_lbl} {_top2_p*100:.1f}%).")
                    final_decision = f"혼합형({_top1_lbl} 우세)"
                else:
                    st.info(f"➡️ PD 하위 집단 예측: **{pred_sub}** ({pred_prob*100:.1f}%)")
                    final_decision = pred_sub

                # 학습기반 cut-off 경고
                sub_cut = None
                if CUTS and isinstance(CUTS, dict):
                    sub_cut = (CUTS.get("step2_cutoff_by_class") or {}).get(pred_sub, None)
                if sub_cut is not None and pred_prob < float(sub_cut):
                    st.warning(f"⚠️ 예측 확률이 학습기반 cut-off({float(sub_cut):.2f}) 미만입니다. '불확실'로 해석/재검 권고")

            # 정상(또는 정상주의)에서도 Step2 참고 표시(요청에 맞춰 간단 버전)
            show_step2_reference = str(final_decision).startswith('Normal')
            if show_step2_reference and model_step2:
                st.info("ℹ️ **임상 참고:** PD 하위집단 모델은 PD 데이터로 학습되었으므로, 정상 케이스에서는 *참고용*으로만 확인하세요.")
                try:
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
                    pairs = sorted(zip(sub_classes, probs_sub), key=lambda x: float(x[1]), reverse=True)
                    if pairs:
                        st.caption("참고 하위집단 확률(상위 2개)")
                        st.write(f"- {pairs[0][0]}: {pairs[0][1]*100:.1f}%")
                        if len(pairs) > 1:
                            st.write(f"- {pairs[1][0]}: {pairs[1][1]*100:.1f}%")
                except Exception:
                    pass

            # 해석 문구
            sex_code = "M" if subject_gender == "남" else "F"
            pos, neg = generate_interpretation(
                prob_normal,
                final_db,
                final_sps,
                float(range_adj),
                float(p_artic),
                float(vhi_total),
                float(vhi_e),
                sex=sex_code
            )
            st.markdown("### 🧾 해석 요약")
            if pos:
                st.markdown("**✅ 긍정적 소견**")
                for r in pos:
                    st.write(f"- {r}")
            if neg:
                st.markdown("**⚠️ 주의/개선 포인트**")
                for r in neg:
                    st.write(f"- {r}")

            # 저장용 데이터 세팅
            patient_info = {
                "name": subject_name,
                "age": int(subject_age),
                "gender": subject_gender,
            }
            analysis = {
                "f0": float(st.session_state.get("f0_mean", 0.0) or 0.0),
                "range": float(range_adj),
                "db": float(final_db),
                "sps": float(final_sps),
                "vhi_total": float(vhi_total),
                "vhi_p": float(vhi_p),
                "vhi_f": float(vhi_f),
                "vhi_e": float(vhi_e),
                "p_pitch": float(p_pitch),
                "p_prange": float(p_prange),
                "p_loud": float(p_loud),
                "p_rate": float(p_rate),
                "p_artic": float(p_artic),
            }
            diagnosis = {
                "final": str(final_decision),
                "normal_prob": float(prob_normal),
            }
            step1_meta = {
                "p_pd": float(p_pd),
                "p_normal": float(p_norm),
                "cutoff": float(pd_cut),
            }
            st.session_state["save_ready_data"] = {
                "patient_info": patient_info,
                "analysis": analysis,
                "diagnosis": diagnosis,
                "step1_meta": step1_meta,
            }
            st.success("✅ 저장/전송 준비 완료! 아래 버튼을 눌러 전송하거나(SQLite/시트) 저장하세요.")

    # 전송/저장 버튼
    st.markdown("---")
    if st.button("☁️ 결과 저장/전송", key="btn_send"):
        if "save_ready_data" not in st.session_state:
            st.warning("먼저 [🚀 진단 결과 확인]을 눌러 결과를 생성하세요.")
        else:
            data = st.session_state["save_ready_data"]
            wav_path = st.session_state.get("current_wav_path")
            ok, msg = send_email_and_log_sheet(wav_path, data["patient_info"], data["analysis"], data["diagnosis"])
            if ok:
                st.success(msg)
            else:
                st.error(msg)
