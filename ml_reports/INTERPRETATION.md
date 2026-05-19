# ML 파이프라인 결과 해석

실행: `python -m ml.run_pipeline train --csv training_data.csv`
산출물: `ml_reports/summary.csv`, `ml_reports/full_report.txt`

---

## 핵심 결과 (5-fold Stratified CV, macro-F1)

| 과제 | feature_set | 최고 모델 | macro-F1 (mean ± sd) | balanced acc |
|------|-------------|-----------|----------------------|--------------|
| Step-1 (PD vs Normal) | **acoustic_only** | logistic_regression | **0.949 ± 0.036** | 0.949 |
| Step-1 (PD vs Normal) | perceptual_only   | random_forest       | 1.000 ± 0.000      | 1.000 |
| Step-1 (PD vs Normal) | combined          | random_forest       | 1.000 ± 0.000      | 1.000 |
| Step-2 (PD 하위유형)  | **acoustic_only** | svm_rbf             | **0.597 ± 0.206**  | 0.644 |
| Step-2 (PD 하위유형)  | perceptual_only   | gradient_boosting   | 0.926 ± 0.148      | 0.933 |
| Step-2 (PD 하위유형)  | combined          | random_forest       | 0.920 ± 0.160      | 0.933 |

---

## 해석 — `docs/CODE_REVIEW.md` 의 C1 (라벨-피처 순환) 확인

### Step-1 (PD vs Normal) — 양호

- **acoustic_only macro-F1 = 0.95** → 객관 음향 지표만으로도 PD/Normal
  이항 분류가 가능. 실용 수준.
- perceptual_only 가 1.000 인 것은 청지각 평가가 PD 환자에서 거의 모두
  비정상으로 표시되기 때문 — 데이터 라벨링이 청지각 기반이므로 자연스러움.

### Step-2 (PD 하위유형) — **순환 확인**

- **acoustic_only macro-F1 = 0.50–0.60** (1/3 = 0.33 chance level 보다 약간
  높은 수준)
- **perceptual_only macro-F1 = 0.93** (거의 완벽)
- 즉, 음향 지표만으로는 PD 하위유형 (Articulation / Intensity / Rate) 을
  잘 구분하지 못하며, 분류 성능을 견인하는 것은 청지각 점수
  (`P_Loudness, P_Rate, P_Artic`) — 그러나 라벨 자체가 이 청지각 점수의
  최대값으로 정해진 항목이므로 **순환 의존성**.

→ 학술 발표·논문에는 acoustic_only 의 0.5–0.6 수치를 보고해야 함.
   현재 `app.py` 가 학습/저장 후 추론에 사용하는 모델 (`P_*` 포함) 은
   부풀려진 정확도를 보고하고 있음.

### 보조 관찰

- Step-2 의 PD_Rate 클래스 (n=6) 가 매우 적어 모든 모델에서 recall 이
  0.67 수준에서 막힘. 추가 표본 수집 필요.
- Step-2 의 perceptual_only 가 combined 보다 약간 높은 fold 가 있는 것은
  acoustic 노이즈가 클래스 분리에 방해가 되기 때문 (특히 `gradient_boosting`).

---

## 권장 후속 조치

1. `app.py` 의 학술 보고용 정확도 표시는 acoustic_only 값으로 교체
   (혹은 두 값 모두 병기 + 한계 명시).
2. PD_Rate 표본 확대 (현 n=6 → 최소 15–20 권장).
3. Step-2 의 청지각-기반 라벨을 객관적 음향 클러스터링으로 재정의하는
   것을 장기 검토 (semi-supervised k-means + Praat feature space).
4. `ml/run_pipeline.py` 의 acoustic_only logistic_regression 모델을
   파일로 직렬화 (`joblib.dump`) 하여 운영 환경에서 재사용 가능하게 만들기.
