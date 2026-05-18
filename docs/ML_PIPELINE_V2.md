# ML Pipeline v2 — 사용법

새로 추가된 `ml/` 모듈 사용법입니다. 원본 `app.py` 는 그대로
동작하며, 본 모듈은 옆에서 독립적으로 실행됩니다.

## 설치

```bash
pip install -r requirements.txt
pip install -r requirements_ml.txt    # 추가 ML 의존성 (imbalanced-learn)
```

## 전체 CV 평가 보고서 생성

```bash
python -m ml.run_pipeline train --csv training_data.csv
```

다음을 실행합니다:

- **Task 1 (Step 1)**: PD vs Normal (이진 분류)
- **Task 2 (Step 2)**: PD_Intensity vs PD_Rate vs PD_Articulation
  (3-class, PD 만)

각 task 에 대해 세 가지 feature set 으로 모두 학습합니다:

- `acoustic_only` — F0, Range, dB, SPS, 성별. **CODE_REVIEW.md C1 의 순환
  의존성을 피하는 set 입니다. 이 값이 임상 보고용 기준이 되어야
  합니다.**
- `perceptual_only` — 청지각 5종 + VHI 4종.
- `combined` — 모두.

모델은 다음 5종 비교:

| Model | 비고 |
|---|---|
| LogisticRegression | 원본 app.py 의 Step1 모델 (재현용) |
| LinearDiscriminantAnalysis | 원본 app.py 의 Step2 모델 (재현용) |
| RandomForest | 비선형, feature importance 제공 |
| SVM (rbf) | 비선형, 작은 표본에 강함 |
| GradientBoosting | 비선형, 표 형식 데이터에 강함 |

**결측치 처리**: `SimpleImputer(strategy="median")` 을 Pipeline 안에서 fold 별
fit. VHI 결측은 0 이 아닌 NaN 으로 유지됩니다.

**클래스 불균형 처리**: `class_weight="balanced"` + SMOTE (k_neighbors=2,
가장 작은 클래스에 맞춤). PD_Rate 5명도 학습에 들어옵니다.

**평가 지표**:

- Macro-F1 (평균 ± SD, fold별 값)
- Balanced accuracy
- 클래스별 precision/recall/F1
- Confusion matrix (out-of-fold 예측 기반)

결과는 콘솔 출력 + `ml_reports/summary.csv` 에 저장됩니다.

## 단일 WAV 에서 피처 추출

```bash
python -m ml.run_pipeline extract --wav subject.wav
```

다음을 출력합니다 (JSON):

- F0 mean/SD/range/min/max
- Jitter (local, rap)
- Shimmer (local, apq5)
- HNR
- Intensity mean/SD
- F1/F2/F3 mean
- MFCC 1–13 mean/SD
- Duration, voiced fraction

`training_data.csv` 의 4-피처 (F0/Range/dB/SPS) 보다 약 30 개 추가
피처를 제공합니다. **추후 새로 수집되는 WAV 에 대해 이 함수를
사용하면 자가 보고 항목(SPS, 청지각, VHI) 없이도 모델 입력이
가능합니다.**

## 의도된 사용 흐름

1. 위 CV 보고서 실행 → `summary.csv` 확인
2. `acoustic_only` 의 성능이 임상 의사결정 지원에 충분한지 판단
3. 부족하다면: 라벨 재정의 또는 데이터 수집 확대
4. 충분하다면: `fit_final_model()` 로 전체 데이터 재학습 후 pickle
   저장, `app.py` 의 inference 경로에 swap-in
