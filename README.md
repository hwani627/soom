# 🌙 Soom CPAP Quality Analyzer

가정용 CPAP(ResMed AirSense 10 AutoSet) raw CSV 데이터를 업로드해서 **시계열 그래프**를 보고, **Apnea/Hypopnea를 직접 검출**하여 SleepHQ 라벨과 일치율을 비교하고, **학술 근거 기반 사용 품질 등급**을 확인하는 시연용 데모 앱입니다.

본 도구는 PSG급 정확도를 지향하는 **Soom CPAP 플랫폼**의 proof-of-concept입니다.

## 📦 기능

- 7일치 SleepHQ 통합 데이터 (long-format raw + 이벤트 라벨 + Sleep Stage)
- 12채널 시계열: Breathing / Pressure / EPAP / Leak / Flow Limit / Snore / SpO₂ / Pulse Rate / Movement / Tidal Volume / Resp Rate / Minute Vent (+ Sleep Stage)
- Apnea/Hypopnea 검출 2종 (AASM, ResMed) + 임계값 슬라이더
- **timestamp 단위** SleepHQ 라벨 일치율 (Precision/Recall/F1)
- 7지표 + 종합 품질 등급 카드 (AHI, Leak, Usage, Pressure, ODI 3%, T90, Lowest SpO₂)

## 🚀 빠른 시작

### 로컬 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```

브라우저에서 http://localhost:8501 자동 오픈.

### 테스트

```bash
pip install pytest
pytest tests/ -v
```

### Streamlit Community Cloud 배포

1. https://share.streamlit.io 접속 후 GitHub 로그인
2. **"New app"** → Repo `hwani627/soom`, branch `main`, main file `app.py`
3. **"Deploy!"** → 약 2~3분 후 배포 완료
4. 이후 `git push` 시 자동 재배포

## 📁 폴더 구조

```
soom/
├── app.py                   # Streamlit 진입점
├── src/                     # 핵심 모듈 (UI와 분리, 향후 본 플랫폼 이식)
│   ├── loader.py            # CSV 파싱
│   ├── detector.py          # Apnea/Hypopnea 검출 (AASM/ResMed)
│   ├── evaluator.py         # 품질 등급 + SleepHQ 비교
│   └── plotting.py          # Plotly 차트
├── CAPA_Data/               # SleepHQ 7일치 통합 export (long format)
│   ├── sleephq_rawdata_*.csv         # 12채널 raw (long format)
│   ├── sleephq_events_AHI_*.csv      # CA/H/RERA 이벤트 (timestamp 정확)
│   └── sleephq_sleep_stages_*.csv    # 수면 단계 transitions
├── tests/                   # pytest 단위 테스트
├── docs/specs/              # 설계 문서
├── .streamlit/config.toml   # 테마·서버 설정
├── requirements.txt
├── runtime.txt              # python-3.11
└── README.md
```

## 🔬 알고리즘

### Apnea / Hypopnea 검출

| 모드 | Apnea | Hypopnea | Baseline |
|---|---|---|---|
| **AASM** | amplitude ≥90% 감소가 ≥10s | ≥30% 감소 (SpO₂ 검증 불가) | 직전 2분 mean |
| **ResMed** | <25% baseline가 ≥10s | <50% baseline가 ≥10s | 100s rolling 60th pct |

### 품질 등급 카드

| 카드 | 기준 | 출처 |
|---|---|---|
| AHI | <5 정상 / 5–15 경증 / ≥15 중등 이상 | AASM 2012 (Berry et al.) |
| Leak (95p) | <24 L/min | ResMed Clinical Guideline |
| Usage | ≥4 h/night | CMS Medicare 2008 (CAG-00093R2) |
| Pressure (95p) | / max < 0.9 | ResMed AutoSet manual |
| **ODI 3%** | <5 / 5–15 / ≥15 | AASM 2012 |
| **T90** | <1% 정상 / <5% 경계 / ≥5% 위험 | Punjabi NM, Sleep Med 2009 |
| **Lowest SpO₂** | ≥90% 정상 / 85–89% 경계 / <85% 저산소 | PSG standard |
| 종합 | AND 게이트 (최하위 등급) | — |

## ⚠️ 한계 (현재 버전)

- **CA/OA 구분 불가**: ResMed의 FOT(Forced Oscillation Technique) 채널이 export CSV에 포함되지 않음. SleepHQ 라벨에는 CA/H가 분리되어 있으나 본 검출기 자체는 통합 Apnea만 산출.
- **Sleep Stage 인코딩 의미 미확정**: 데이터 파일의 stage 값(2/3/4/5)에 대한 wearable별 매핑은 추후 확정 필요.

## 📚 학술 근거 출처

- Berry RB et al. *J Clin Sleep Med* 2012; 8(5): 597–619 (AASM 스코어링)
- ResMed AirSense 10 AutoSet Algorithm White Paper
- Farré R et al. *Eur Respir J* 1998 (FOT)
- CMS Decision Memo 2008 (CAG-00093R2) (Medicare CPAP Adherence)
- ResMed Clinical Guideline (Mask Leak)

## 🛣️ 로드맵

- v1.1: 다중 날짜 추세 대시보드
- v1.2: 2개 날짜 비교 모드
- v1.3: SpO₂ / Pulse Rate 외부 동기화
- v2.0: AI 모델 통합 (CA/OA 추정 등) — 별도 백엔드(HF Spaces) 추가
- v2.1: 본 PSG급 플랫폼 본 코드로 이식

## 📝 라이선스 / 면책

본 데모는 **시연·연구 목적**이며 임상 진단을 대체하지 않습니다. 데이터는 익명·비식별화된 데모 데이터입니다.
