# 🎙️ 자발화 분석 도구 (Spontaneous Speech Analyzer)

언어치료 자발화 분석을 위한 **로컬 데스크톱 도구** (Streamlit 단일 앱).
음성 녹음 → 듀얼 전사 → MLU/조음 분석 → 보고서.

대상: 아동 + 성인 (언어발달지연/장애, 실어증). 환자 음성은 로컬에만 저장됩니다.

## 분석 모드

| 모드 | 입력 | 산출 |
| --- | --- | --- |
| 📝 언어 분석 | 텍스트 또는 음성 | MLU-w, MLU-m, TTR, NDW, TNW, 문법형태소 분포 |
| 🔊 조음 분석 | 음성 필수 | 컨퓨전 매트릭스, PCC, 음소별 정확도, 위치별 오류 |
| 🎯 통합 분석 | 음성 필수 | 언어 + 조음 + 종합 보고서 |

## 구현 현황 (마일스톤)

- [x] **M1** — 메인 페이지 + 언어 분석(텍스트 입력)
- [x] **M2** — 음성 업로드 + Whisper 전사 → 발화별 화자 지정(아동/치료사/제외) → 아동만 분석
- [x] **M3** — GPT-4o audio 산출형 전사 + 듀얼 검수 + 조음 분석(PCC·컨퓨전 매트릭스·위치별 오류)
- [x] **M4** — 통합 분석 페이지 (언어+조음 탭 + 종합 보고서)
- [x] **M5** — 인사이트 레이어 (APAC 분류 기반 LLM 임상 코멘트)
- [ ] M6 — Tauri 패키징 (.exe)

## 설치 & 실행 (Windows, Python 3.11 권장)

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

copy .env.example .env          # OPENAI_API_KEY 입력 (음성 모드용)
streamlit run app.py
```

브라우저에서 메인 페이지 → 모드 카드 선택.

> 텍스트 언어 분석·산출형 직접 입력은 OpenAI API 키 없이도 동작합니다.

## 접근 비밀번호 + 공용 키 (운영자 부담 모델)

닫힌 그룹(수업 학생·동료)을 위한 권장 구성입니다.
- **비밀번호 잠금**: `APP_PASSWORD`를 설정하면 비밀번호를 아는 사람만 접속 가능(무단 사용·비용 사고 방지).
- **공용 OpenAI 키**: `OPENAI_API_KEY`를 설정하면 사용자는 키 입력 없이 바로 사용. 사용료는 운영자 계정에 청구.
  → **OpenAI 대시보드에서 월 사용 한도(spend limit)를 꼭 설정**하세요.
- 두 값 모두 **미설정 시**: 비밀번호 없이 열리고, 사용자가 사이드바에 본인 키를 입력하는 방식으로 동작(로컬 개발).
- `ffmpeg`는 `imageio-ffmpeg`로 자동 포함되어 **별도 설치가 필요 없습니다**(mp3·m4a·wav 모두 동작).

로컬에서 비밀번호/키를 쓰려면 `.streamlit/secrets.toml.example`을 `.streamlit/secrets.toml`로 복사해 채웁니다.

## 웹앱 배포 (다른 사람도 브라우저로 사용)

이미 Streamlit 앱이라 바로 웹 배포가 가능합니다.

**Streamlit Community Cloud (가장 쉬움, 무료)**
1. 이 폴더(`spontaneous-speech-analyzer/`) 내용을 **전용 GitHub 저장소의 루트**로 올립니다.
   (Streamlit Cloud는 저장소 루트의 `requirements.txt`를 사용하므로, 하위 폴더보다 전용 repo가 깔끔합니다.)
2. share.streamlit.io → New app → 저장소 선택 → Main file: `app.py` → Deploy
3. **App settings → Secrets**에 아래를 붙여넣기:
   ```toml
   APP_PASSWORD = "공유할-비밀번호"
   OPENAI_API_KEY = "sk-..."
   ```
4. 배포된 `...streamlit.app` 링크 + 비밀번호를 그룹에 공유. 사용자는 키 입력 불필요.

> 무료 플랜은 일정 시간 미사용 시 앱이 잠들고, 다음 접속자가 깨우는 데 30초~1분 걸립니다(데이터 손실 없음).
> 항상 켜둬야 하면 Render/VPS 등 상시 가동(월 $5~7)으로 옮기면 됩니다.

> ⚠️ 환자 음성은 업로드 시 서버와 OpenAI로 전송됩니다. 보호자 동의·기관 정책을 확인하세요.

## 디렉토리 구조

```
spontaneous-speech-analyzer/
├── app.py                       # 메인 (홈 + 모드 카드)
├── pages/
│   ├── 1_📝_언어_분석.py        # 언어 분석 (M1: 텍스트 입력)
│   ├── 2_🔊_조음_분석.py        # 조음 분석 (M3 예정)
│   └── 3_🎯_통합_분석.py        # 통합 분석 (M4 예정)
├── modules/
│   ├── __init__.py
│   ├── morpheme.py              # kiwipiepy → MLU/TTR/품사/문장유형
│   ├── transcription.py         # Whisper 목표어 + GPT-4o audio 산출형 전사
│   ├── g2p.py                   # g2pkk 래퍼 (목표어 → 발음형)
│   ├── jamo_split.py            # 초성/중성/종성 분리
│   ├── articulation.py          # 컨퓨전 매트릭스 + PCC + 위치별 오류
│   ├── insights.py              # APAC 기반 LLM 임상 코멘트
│   └── shared_ui.py             # 모드 간 공유 UI 컴포넌트
├── data/
│   └── few_shot_examples.json   # 산출형 전사 few-shot
├── .env.example
├── requirements.txt
└── README.md
```

## 지표 정의 (언어 분석)

낱말(word) = **내용어 = 체언 + 용언 + 수식언 + 독립언** (임상 자발화 분석 정답지 양식).

- **체언**: 명사(NNG/NNP/NNB/NR), 대명사(NP)
- **용언**: 동사(VV/VX), 형용사(VA/VCN) — *기본형으로 표기*
- **수식언**: 부사(MAG/MAJ), 관형사(MM)
- **독립언**: 감탄사(IC)
- *지정사 '이다'(VCP)는 서술격조사로 보아 낱말에서 제외*

- **MLU-w** (평균 낱말 길이) = 총 낱말 수 / 발화 수
- **MLU-m** (평균 형태소 길이) = 총 형태소 수 / 발화 수
- **TNW** = 총 낱말 수, **NDW** = 서로 다른 낱말 수, **TTR** = NDW / TNW
- **의미 영역 세분화** = 명사·대명사·동사·형용사·부사·관형사·독립언 (총 빈도 + type 수)
- **문법형태소 세분류** = 조사 / 연결어미 / 어말어미 / 선어말어미 / 전성어미
  - *피동·사동 접사는 형태소 분석기가 어간에 병합 → 자동 분리 불가(임상가 검수)*
- **문장유형 (자동 추정)** = 단문 / 이어진문장(연결어미) / 안긴문장(전성어미·관형절)
  - *관형 수식·인용절 등은 정확도가 낮아 임상가 검수 필요*

> 정확한 지표를 위해 **표준어 정규화 + 마디(반복·수정·간투사) 제외** 전사를 권장합니다.
> kiwipiepy 자동 분석은 임상 정답지의 약 1.5배까지 과다 집계될 수 있어, 임상가 검수가 필요합니다.
