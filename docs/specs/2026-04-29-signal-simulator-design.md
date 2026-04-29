# Signal Simulator — Design Spec

- **Date**: 2026-04-29
- **Author**: YoungHwan Choi
- **Status**: Draft (awaiting user review)
- **Spec reference**: `3.개발기술/03_신호처리_사양서.md` v0.4 §0.6
- **Related code**: `src/signal_processing/generator.py`, `scripts/demo_signal_pipeline.py`
- **Related spec**: `docs/specs/2026-04-27-edf-viewer-design.md` (sibling Streamlit page)

---

## §1. 개요 (Purpose)

`generator.py`의 `ScenarioConfig` 합성 엔진을 인터랙티브 Streamlit 페이지로 감싸,
사용자가 슬라이더와 위젯으로 양압기 압력 raw 신호의 임상·기계적 특성을 실시간 조작하고,
결과를 시각적으로 확인하며, **외부 독립 알고리즘 검증용 CSV 세트**로 내보낼 수 있도록 한다.

### 해결하려는 문제

1. 현재 `demo_signal_pipeline.py`는 `default_demo_scenario()` 하드코딩 단일 시나리오만 생성 — 다양한 임상 케이스 탐색 불가.
2. 외부 분석 알고리즘 검증을 위한 표준화된 ground-truth 라벨 데이터셋이 없음.
3. 임상의·검증 엔지니어가 코드 수정 없이 시나리오를 만들 방법이 없음.

### Non-goals

- 실시간 스트리밍 (전체 세션 일괄 생성만)
- AASM 자동 채점 UI (외부 알고리즘이 담당)
- 환자 phenotype preset (Tier 3, 본 spec 범위 밖)

### 시나리오 범위 (Tier 1 + Tier 2)

| Tier | 항목 |
|---|---|
| Tier 1 (기존 유지) | RR, TV, base_pressure, 심박수, cardiogenic 진폭, 측정 노이즈, intentional leak, OA·CA·hypopnea·snore 횟수/지속시간 |
| Tier 2 (신규 추가) | 50Hz 전원 간섭, MA(Mixed Apnea), unintentional leak (constant/ramp/burst), cough, EPR (호기 압력 완화) |
| Tier 3 (제외) | CSR 합성, RERA, body movement, ramp-up, APAP, sleep stage |

---

## §2. 아키텍처 & 컴포넌트

### 2.1 파일 구조 (신규 추가)

```
soom_Project/
├── pages/                                   # Streamlit multi-page 규약
│   └── 2_📈_Signal_Simulator.py            # 신규 (메인 페이지)
├── src/
│   ├── signal_processing/
│   │   ├── generator.py                    # 기존 — Tier 2 항목 추가 확장
│   │   └── ...
│   └── simulator/                          # 신규 (UI 보조 로직)
│       ├── __init__.py
│       ├── ui_state.py                     # 슬라이더 → ScenarioConfig 변환
│       ├── randomizer.py                   # 🎲 Randomize 로직
│       ├── exporter.py                     # CSV/JSON/ZIP 패키징
│       └── plotting.py                     # plotly 멀티패널
└── tests/
    └── test_simulator.py                   # 신규
```

### 2.2 컴포넌트 책임 분리

| 모듈 | 입력 | 출력 | 책임 |
|---|---|---|---|
| `pages/2_📈_Signal_Simulator.py` | 사용자 위젯 입력 | Streamlit UI | 위젯 배치, 콜백 연결, 레이아웃 |
| `simulator/ui_state.py` | 위젯 dict | `ScenarioConfig` | 사용자 입력 → 합성 엔진 입력 변환 |
| `simulator/randomizer.py` | 임상 정상 범위 | dict (슬라이더 값) | 🎲 정상 분포 샘플링, 자동 랜덤 메타 생성 |
| `simulator/exporter.py` | signals + GT + cfg | bytes (zip) | CSV 2개 + JSON 1개 + README → ZIP |
| `simulator/plotting.py` | signals + events | plotly Figure | 멀티패널 (이벤트 음영 오버레이) |
| `signal_processing/generator.py` | `ScenarioConfig` | (signals dict, GroundTruth) | 기존 — Tier 2 추가만 |

UI 모듈은 generator를 모르고, generator는 UI를 모름 (단방향 의존, 테스트 가능성).

### 2.3 `ScenarioConfig` Tier 2 확장

`generator.py`의 `ScenarioConfig` 데이터클래스에 다음 필드 추가 (모두 기본값 무동작 — backward compatible):

```python
# Tier 2 추가
ie_ratio: float = 1.0
    # 흡기 시간 / 호기 시간. dataclass 기본 1.0 (= 현재 사인파, backward compat).
    # UI 슬라이더 기본은 0.5 (1:2 정상 성인) — UI가 cfg 빌드 시 명시적으로 설정.
mixed_apneas: list[tuple[float, float, float]] = field(default_factory=list)
    # (start_s, total_duration_s, central_fraction 0~1)
unintentional_leak_lpm: float = 0.0
unintentional_leak_profile: Literal["constant", "ramp", "burst"] = "constant"
cough_events: list[tuple[float, float]] = field(default_factory=list)
    # (start_s, peak_amplitude_cmh2o)
power_line_50hz_amplitude_cmh2o: float = 0.0  # 0이면 비활성
epr_enabled: bool = False
epr_relief_cmh2o: float = 0.0  # 호기 시 압력 ↓ (0~3)
```

`synthesize_session()` 본문에 위 변수 처리 분기 추가.
- `ie_ratio` 지원을 위해 `_breath_flow`를 비대칭 파형(흡기 ie_ratio×T, 호기 (1−ie_ratio)×T)으로 일반화. 단 TV 적분이 보존되도록 진폭 보정.
- 기본값 `ie_ratio=1.0` 시 현재 사인파와 동등 — 기존 17개 테스트 통과 보존.

### 2.4 데이터 의존성 (단방향)

```
사용자 위젯 → ui_state → ScenarioConfig
                            ↓
                    generator.synthesize_session()  ← randomizer (시드·jitter 주입)
                            ↓
                    (signals, GroundTruth)
                            ├→ plotting → 화면
                            └→ exporter → ZIP 다운로드
```

---

## §3. UI 레이아웃

### 3.1 페이지 구조 (Streamlit `wide` mode)

```
┌─────────────────────────────────────────────────────────────────┐
│  📈 Signal Simulator — 양압기 압력 신호 합성 도구                │
├──────────────────┬──────────────────────────────────────────────┤
│  [사이드바]      │  [메인 영역]                                  │
│                  │  📊 신호 그래프 (plotly, 인터랙티브)          │
│ ▶ 세션           │  ┌─────────────────────────────────────┐    │
│ ▶ 호흡 패턴      │  │  Pressure (cmH₂O)  [이벤트 음영]    │    │
│ ▶ 압력 시스템    │  ├─────────────────────────────────────┤    │
│ ▶ 이벤트         │  │  Flow ground truth (L/min)          │    │
│ ▶ 심혈관         │  └─────────────────────────────────────┘    │
│ ▶ 노이즈/누설    │  [▼ 고급 보기 (RPM, 4-branch BPF)]          │
│                  │                                              │
│ ─────────────    │  📋 자동 랜덤 메타 (read-only)               │
│ 🎲 Randomize    │  📥 Export — [⬇ Download ZIP]                │
│ ↻ Reset         │  ▶ Advanced: 이벤트 테이블 직접 편집           │
└──────────────────┴──────────────────────────────────────────────┘
```

### 3.2 사이드바 슬라이더 그룹 (총 ~18개, expander로 접힘)

| 그룹 | 위젯 | 범위 | 기본값 |
|---|---|---|---|
| 세션 | duration_s | 60~1800 | 300 |
| 호흡 패턴 | rr_bpm / tv_ml / ie_ratio | 8~30 / 200~800 / 0.3~1.0 | 15 / 500 / 0.5 |
| 압력 시스템 | base_pressure / epr_enabled / epr_relief | 4~20 / on-off / 0~3 | 9.5 / off / 1.5 |
| 이벤트 (개수) | OA / CA / MA / Hypopnea / Snore / Cough | 0~20 / 0~10 / 0~5 / 0~20 / 0~10 / 0~10 | 2 / 1 / 0 / 1 / 1 / 0 |
| 이벤트 (평균 지속시간) | OA·CA·hypopnea 평균 dur | 10~60 s | 18 / 15 / 20 |
| 심혈관 | hr_bpm / cardiogenic_amp | 40~100 / 0~0.5 | 70 / 0.25 |
| 노이즈/누설 | meas_noise / 50Hz_on / unintentional_leak | 0~0.2 / on-off / 0~30 | 0.05 / off / 0 |

`fs_hz`는 100 Hz 고정 (read-only 표시) — 사양서 §3.2 정합.

### 3.3 🎲 Randomize 동작 정의

- 클릭 시 위 슬라이더들을 **임상 정상 범위 내**에서 균등/정규분포 샘플링 후 즉시 페이지 재실행.
- **항상 자동 랜덤 (UI 노출 X, read-only 메타로만 표시)**: 이벤트별 정확한 시점, 지속시간 jitter, hypopnea severity, RNG seed.
- **🎯 Pin seed** 체크박스: 켜두면 같은 슬라이더 값으로 같은 결과 재현.

### 3.4 메인 영역 패널

1. **신호 그래프** — plotly 2단(기본) / 4단(고급 보기 토글)
   - 이벤트 음영: OA=빨강, CA=파랑, MA=보라, Hypopnea=노랑, Snore=초록, Cough=주황
   - hover 툴팁: (시작·종료, 지속, severity)
2. **자동 랜덤 메타 패널** — 사용자 미제어 랜덤 결과 표시
3. **Export 버튼** — `soom_simulator_{YYYY-MM-DD_HHMMSS}_{seed}.zip`
4. **Advanced: 이벤트 테이블** — `st.data_editor`, 자동 배치 결과 직접 수정 가능

---

## §4. 데이터 흐름 · 캐시 · Export 명세

### 4.1 페이지 재실행 시 데이터 흐름

```
[Streamlit rerun]
  ① ui_state.collect_widgets() → dict
  ② ui_state.build_scenario_config(dict, seed) → ScenarioConfig
  ③ generate_signals_cached(cfg, seed)           ← @st.cache_data
       ↓ (캐시 미스 시에만 계산)
  ④ (signals dict, GroundTruth) 반환
  ⑤ plotting.build_figure(signals, gt, show_advanced) → plotly Fig
  ⑥ st.plotly_chart + 메타 패널 + Export 버튼 등록
```

캐시 키: `ScenarioConfig` 전체 dict + seed 해시. 캐시 한도: `max_entries=20` (LRU).

### 4.2 성능 가드

- `duration > 600s`(10분) 또는 `fs > 200Hz` 일 때 사이드바 상단에 ⚠️ 경고 안내.
- `duration > 1800s` 슬라이더 hard cap.

### 4.3 시드 처리

- **Pin seed OFF (기본)**: `seed = int(time.time() * 1000) % 2**32` — 매번 새 시드 → cache miss 보장.
- **Pin seed ON**: `seed = st.session_state.get("pinned_seed", 42)` — 슬라이더 같으면 cache hit.

### 4.4 Export ZIP 명세

**파일명**: `soom_simulator_{YYYY-MM-DD_HHMMSS}_{seed}.zip`

**ZIP 내부 구조**:
```
signal.csv         — UTF-8, LF, 헤더 1행
ground_truth.csv   — 동일
metadata.json      — UTF-8, indent=2
README.txt         — 외부 알고리즘 사용자용 안내
```

#### signal.csv 스키마 (행 수 N = duration × fs)

| 열 | 단위 | 비고 |
|---|---|---|
| `time_s` | s | 0부터, 1/fs 간격 |
| `pressure_cmh2o` | cmH₂O | **외부 알고리즘 입력 (핵심)** |
| `flow_lpm` | L/min | ground truth (검증용, 알고리즘 미사용 권장) |
| `flow_patient_lpm` | L/min | leak 제외 환자 성분 |
| `blower_rpm` | RPM | Phase 0 알고리즘이 활용 가능 |

#### ground_truth.csv 스키마

| 열 | 비고 |
|---|---|
| `event_type` | OA / CA / MA / hypopnea / snore / cough |
| `start_s` | 이벤트 시작 |
| `end_s` | 이벤트 종료 |
| `duration_s` | end − start |
| `severity` | hypopnea만 채워짐, 그 외 빈 문자열 |
| `metadata` | JSON 직렬화 (MA의 central_fraction 등) |

#### metadata.json 스키마

```json
{
  "schema_version": "1.0",
  "generated_at": "2026-04-29T10:30:45+09:00",
  "seed": 42,
  "fs_hz": 100.0,
  "duration_s": 300.0,
  "scenario": {
    "rr_bpm": 15.0, "tv_ml": 500.0, "base_pressure_cmh2o": 9.5,
    "intentional_leak_lpm": 24.0, "hr_bpm": 70.0,
    "cardiogenic_amplitude_cmh2o": 0.25,
    "measurement_noise_std_cmh2o": 0.05,
    "epr_enabled": false, "epr_relief_cmh2o": 0.0,
    "power_line_50hz_amplitude_cmh2o": 0.0,
    "unintentional_leak_lpm": 0.0
  },
  "event_summary": {
    "OA": 2, "CA": 1, "MA": 0, "hypopnea": 1, "snore": 1, "cough": 0
  },
  "soom_version": "git short hash",
  "spec_reference": "03_신호처리_사양서.md v0.4 §0.6"
}
```

#### README.txt (외부 알고리즘 사용자용)

```
soom_Project Synthetic CPAP Pressure Signal — Validation Dataset
================================================================
1. Read signal.csv into your analysis tool
   → Use 'pressure_cmh2o' column as the input signal (fs given in metadata.json)
2. Run your apnea/hypopnea detector on that single column
3. Compare your detections with ground_truth.csv
   (event_type, start_s, end_s) tuples are the labels
4. metadata.json describes the synthesis parameters

Note: 'flow_lpm' / 'flow_patient_lpm' are GROUND TRUTH only.
A real CPAP device with a single ΔP sensor cannot directly measure flow.
```

---

## §5. 오류 처리 · 테스트 전략

### 5.1 오류 처리 매트릭스

| 시나리오 | 감지 위치 | 처리 |
|---|---|---|
| 이벤트 합산 시간 > duration | `ui_state.build_scenario_config` | `st.warning` + 자동 클리핑 (진행 허용) |
| 이벤트 시점 겹침 | `randomizer._allocate_events` | 자동 재배치 (max 50 시도), 실패 시 `st.warning` |
| duration > 1800s | 슬라이더 hard cap | 위젯 자체에서 차단 |
| TV·RR 곱 비현실적 (MV > 25 L/min) | `ui_state` 검증 | `st.warning`, 진행 허용 |
| Hypopnea severity = 0 (≈ apnea) | `ui_state` 검증 | `st.info` 권고 |
| `synthesize_session` 내부 예외 | 페이지 try/except | `st.error(traceback)`, 마지막 성공 시나리오 유지 |
| Advanced 테이블 잘못된 입력 | `data_editor` 콜백 | 무효 행 빨강 표시 + 시그널 생성에서 제외 |
| Export ZIP 생성 실패 | `exporter.build_zip` 예외 | `st.error("Export 실패: {reason}")` |
| 캐시 키 hashable 아님 | `@st.cache_data` 직렬화 | cfg는 `dataclass(frozen=True)` 또는 dict 변환 후 캐시 |

검증 정책: soft warning(진행 허용) vs UI 위젯 min/max(hard block) — 코드 추가 block 없음.

### 5.2 테스트 전략

#### 신규 `tests/test_simulator.py` (12~15건)

```python
class TestUIState:
    test_widget_dict_to_scenario_config_roundtrip
    test_event_count_to_schedule_no_overlap
    test_event_count_to_schedule_clipped_when_too_long
    test_ie_ratio_changes_inspiration_duration
    test_advanced_table_overrides_count_based_schedule

class TestRandomizer:
    test_randomize_within_clinical_ranges
    test_pinned_seed_produces_identical_config
    test_event_jitter_respects_duration_cap

class TestExporter:
    test_zip_contains_three_files_plus_readme
    test_signal_csv_row_count_matches_duration_fs
    test_ground_truth_csv_event_count_matches_gt
    test_metadata_json_schema_v1
    test_csv_readable_by_pandas_roundtrip       # 외부 호환성

class TestGeneratorTier2Extensions:
    test_mixed_apnea_central_then_obstructive
    test_unintentional_leak_ramp_increases_total_flow
    test_50hz_interference_appears_in_spectrum
    test_epr_drops_pressure_during_expiration
    test_cough_event_creates_pressure_spike
```

#### `tests/test_signal_processing.py` 기존 17건 그대로 통과 (backward compat 검증).

### 5.3 수동 Smoke Test 체크리스트 (UI)

1. 페이지 로드 성공
2. 모든 슬라이더 움직임 → 그래프 갱신 (<500 ms @ 5 min·100 Hz)
3. 🎲 Randomize 클릭 → 슬라이더·그래프·메타 갱신
4. ZIP 다운로드 → 압축 풀어 외부 Python으로 `pandas.read_csv` 성공
5. ZIP의 ground_truth와 EDF Viewer 페이지에서 cross-check
6. duration=1800 s — 경고 표시 + 결국 동작
7. duration=60 s — 빈 cycle 없이 그래프
8. Tier 2 옵션 모두 활성 → ZIP 정상

### 5.4 의존성

기존 stack만 사용: `streamlit`, `numpy`, `pandas`, `scipy`, `plotly`, `pyedflib`. 신규 의존성 없음.

---

## §6. 참고 자료

- AASM Manual for the Scoring of Sleep v2.6 (2023)
- Berry RB et al. *J Clin Sleep Med* 2012;8(5):597-619
- Ayappa I et al. *Chest* 1999;116(3):660-6 (Cardiogenic Oscillation)
- 사내 사양서: `3.개발기술/03_신호처리_사양서.md` v0.4 §0.6
- 자매 spec: `docs/specs/2026-04-27-edf-viewer-design.md`

---

## §7. 외부 검증 워크플로우 가이드

ZIP 산출물을 받은 외부 알고리즘 검증자의 권장 워크플로우:

```python
import pandas as pd, json, zipfile

with zipfile.ZipFile("soom_simulator_2026-04-29_103045_42.zip") as z:
    z.extractall("dataset/")

sig  = pd.read_csv("dataset/signal.csv")
gt   = pd.read_csv("dataset/ground_truth.csv")
meta = json.load(open("dataset/metadata.json"))

# 외부 검출기 호출 — 입력은 pressure_cmh2o 컬럼만
detected = my_detector(sig["pressure_cmh2o"].values, fs=meta["fs_hz"])

# (OA/CA/hypopnea 시점 겹침 매칭으로 sensitivity·PPV 산출)
```
