# pd-voice-diagnosis · 코드 검토 보고서

작성: 2026-05-18 (Voice Lab Hub Phase 4)
대상: `app.py` (1,921줄, Streamlit + parselmouth + scikit-learn)

본 문서는 `app.py` 를 정적 분석한 결과 발견한 방법론적·구현상
이슈를 정리합니다. 일부는 학술 발표·논문 게재 시 reviewer 가 즉시
지적할 수 있는 수준이므로 수정 우선순위가 높습니다.

동작하는 코드를 깨뜨리지 않기 위해 원본 `app.py` 는 수정하지 않고,
옆에 `ml/` 모듈을 신규로 추가했습니다. 교수님 확인 후 마이그레이션
방향을 정하시면 됩니다.

---

## 🔴 Critical (학술적 타당성)

### C1. 라벨-피처 순환 (label leakage via perceptual scores)

`train_models()` 의 Step2 (PD 하위유형) 학습에서 `P_Loudness`,
`P_Rate`, `P_Artic` (청지각 평가 점수) 가 피처로 들어갑니다
(line 1112–1118).

그런데 `training_data.csv` 의 라벨 자체가
**`PD_Intensity / PD_Rate / PD_Articulation`** — 즉 청지각 평가 점수가
가장 두드러진 항목으로 정해진 라벨입니다.

→ 청지각 점수로 정의된 라벨을 다시 청지각 점수로 예측하는
   **순환 의존성** 입니다. 현재 보고되는 분류 정확도는
   부풀려져 있으며, **객관 음향 지표만으로의 분류 성능을
   별도로 보고**해야 합니다.

> 검증 방법: `python -m ml.run_pipeline train` 실행 시
> `feature_set=acoustic_only` 와 `perceptual_only` 의 macro-F1
> 차이를 보면 됩니다. 후자가 훨씬 높다면 위 순환의 증거입니다.

### C2. 학습/평가 분리 부재

`train_models()` 는 `training_data.csv` 전체를 fit 합니다. LOO 가
`compute_cutoffs_from_training()` 에 따로 있지만 cutoff 산출 용도이며,
실제 운영용 모델 평가는 학습 데이터에서 그대로 측정됩니다.

→ 일반화 성능을 알 수 없습니다. 신규 모듈은 **StratifiedKFold (5-fold)
   의 out-of-fold 예측**으로 모든 지표를 산출합니다.

### C3. VHI 결측을 0 으로 채움

`compute_cutoffs_from_training()` (line 700~707) 에서 신규 PD 표본
(PD109~138) 의 비어있는 VHI 컴럼을 0 으로 채웁니다. 그러나 이 값들은
**기록 누락(missing)** 이지 **0 점** 이 아닙니다.

→ "VHI 가 기록되었는가" 자체가 정상/PD 를 구분하는 인공 신호가
   됩니다. 신규 모듈은 NaN 으로 유지하고
   `SimpleImputer(strategy="median")` 을 Pipeline 안에 두어 각 fold 에서
   별도로 fit 합니다.

### C4. SPS 가 음성에서 측정되지 않음

`SPS = user_syllables / selected_window_seconds` — 음절 수는 사용자가
입력합니다 (대본 자동채움 + 수정 가능). 윈도우만 자동 추천이고,
음절 수는 자가 보고입니다.

→ 핵심 피처 하나가 **자가 보고** 라는 뜻입니다. 신규 파이프라인의
   `feature_extraction.extract_features()` 는 voiced fraction 과
   intensity-based VAD 로부터 발화 속도 추정을 제공합니다. 정밀
   음절 인식은 G2P + forced alignment 또는 wav2vec 기반 syllable
   counter 가 횥후 필요합니다.

### C5. dB 절대값이 미보정

intensity 표시에 -50 ~ +50 dB 의 사용자 조정 슬라이더가 있습니다.
캠리브레이션 톤이 없으므로 절대값 비교는 의미가 없습니다.

→ Voice 활동에서는 일반적으로 **샘플 내 상대값**(예: voiced 평균 -
   무음 평균) 으로 정규화합니다. 임상 비교를 원하면 보정 음원
   (피험자마다 동일한 거리에서 1kHz tone 등) 을 함께 수집해야
   합니다.

---

## 🟠 Important (구현 결함)

### I1. `_youden_cutoff` 에 `@st.cache_resource`

NumPy 배열은 해쉬 가능하지 않고, Streamlit 의 `cache_resource` 는 객체
identity 로 캐시합니다. 매 호출마다 신규 배열이 들어오므로 캐시가 안
되거나 경고가 납니다. → 캐시 제거하거나 `cache_data` 로 바꾸세요.

### I2. line 1846 bare `except` (parse 단계 SyntaxError 위험)

선행 `try:` 블록이 line 1844 에서 이미 닫힌 뒤 line 1846 의 `except`
가 홀로 남아 있습니다. 직접 재현 후 확인을 권합니다.

### I3. F0 octave correction 의 ±60% 허용 범위

`lower = median * 0.6`, `upper = median * 1.6` 은 한 옥타브(2 배) 안쪽이라
대부분의 octave-jump 가 그대로 들어옵니다. 신규 모듈은 Praat 의 표준
`To Pitch` 을 그대로 사용합니다.

### I4. `_label_to_diag_and_sub` 의 가변 튜플 반환

대부분의 분기에서 `(diag, sub)` 2-tuple 을 반환하다가 fallback (line 955)
에서 `(None, None, None)` 3-tuple 을 반환합니다. 호출자에서 2-튜플로
언패킹하면 `ValueError` 가 납니다.

### I5. global state (`STATS_STEP1`, `F0Z_STATS`, `MODEL_LOAD_ERROR`)

`train_models()` 안에서 `global` 로 모듈 변수에 대입합니다. Streamlit 의
`cache_resource` 와 결합되면 멀티세션 환경에서 race 가 납니다.
**dataclass 로 묶어 반환**하는 패턴이 안전합니다.

---

## 🟡 Minor (UX / 운영)

- `pd_tool.db` 가 CWD 에 생성됨 → 컨테이너에서 사라짐.
  `~/.pd_tool/db.sqlite` 등 절대 경로 권장.
- `NanumGothic` 폰트가 호스트에 없으면 matplotlib 라벨이 깨짐.
  Streamlit Cloud / Hugging Face Spaces 에서는 `packages.txt` 에
  `fonts-nanum` 추가.
- `find_peaks` import 가 사용되지 않음.

---

## ⏭️ 권장 마이그레이션 순서

1. **`ml/run_pipeline.py train` 으로 신규 CV 보고서 확보** —
   `acoustic_only` 의 macro-F1 이 얼마나 떨어지는지 정량 확인.
2. **임상 합의** — 4-class 라벨이 청지각 점수로 정의된 한, 객관 음향
   만으로 해당 라벨을 맞추는 것은 본질적으로 한계가 있습니다. 라벨
   재정의 (예: UPDRS-Speech 기반, 또는 두 평정자 간 일치도 기반) 를
   검토하시는 것을 권합니다.
3. **app.py 의 Step2 inference 부분만 신규 모듈로 교체** —
   `from ml.training import fit_final_model, model_zoo` 형태로 import.
4. **WAV 입력 시 jitter/shimmer/HNR/MFCC 추가 추출** — 사용자가 입력하지
   않아도 자동으로 채워지는 객관 피처들이 늘어납니다.
