# EDF Viewer (PSG) — Design Spec

- **작성일**: 2026-04-27
- **상태**: 디자인 확정 (구현 전)
- **저장 경로**: `docs/specs/2026-04-27-edf-viewer-design.md`
- **연관 문서**: `docs/specs/2026-04-27-soom-cpap-analyzer-design.md` (기존 v0.2 앱 디자인)

---

## 1. 목적

`CAPA_Data/SC4001E0-PSG.edf`(Sleep-EDF Database 표준 8h PSG)와 같은 EDF 파일을 사이드바에서 선택해 채널별 시계열을 시각적으로 탐색할 수 있는 **독립 EDF 뷰어 페이지**. 검출 알고리즘과 분리된 단순 탐색·교육·디버깅 도구.

기존 Soom CPAP Quality Analyzer(SleepHQ CSV 기반)와 같은 Streamlit 앱 안에서 멀티페이지로 동작하지만, 코드 의존성은 0(완전 분리). 1번 페이지의 회귀 위험을 방지하기 위함.

---

## 2. 범위

### In Scope (v1)

1. `CAPA_Data/*.edf` 자동 스캔으로 EDF 파일 선택
2. EDF 헤더 정보 패널(환자 ID, 시작시간, 길이, 채널/sample rate)
3. 채널 그룹별 빠른 선택 (EEG / EOG / EMG / Resp / SpO2) + 개별 채널 multiselect
4. Hypnogram annotation 자동 페어링 + 차트 하단 색띠 오버레이
5. 자유 zoom 모드 (전체 야간) + 30s epoch 모드 (PSG 표준 리뷰) 토글
6. 자유 zoom 모드에서 plotly_resampler 기반 LTTB 동적 다운샘플링

### Out of Scope (v2 후보)

- 신호 필터 옵션 (band-pass / notch)
- 선택 구간 CSV 내보내기
- 멀티 EDF 좌우 비교 split view
- 사용자 EDF 업로드 (CAPA_Data 자동 스캔만 v1)
- mne 의존 분석 기능 (artefact, ICA 등)

---

## 3. 핵심 결정 사항

| 결정 | 선택 | 이유 |
|---|---|---|
| 호스팅 | Streamlit 멀티페이지 (`pages/2_EDF_Viewer.py`) | 기존 인프라/배포 그대로 재사용 |
| 입력 | 로컬 폴더 자동 스캔 | 기존 CSV 로딩 패턴과 일치, 데모 즉시 가용 |
| 네비게이션 | 자유 zoom + 30s epoch 토글 | 데모성과 임상 워크플로우 동시 충족 |
| EDF 파싱 | `pyedflib` | 가벼움(~2 MB), EDF/EDF+/BDF 지원, annotation 읽음 |
| 다운샘플링 | `plotly-resampler` (FigureResampler) | 동적 LTTB 자동, 직접 구현 불필요 |
| 1번 페이지 코드 재사용 | **하지 않음** | 회귀 위험 0, 책임 분리 명확 |

---

## 4. 시스템 아키텍처

### 4.1 파일 구조 (변경/신규만)

```
soom_Project/
├── pages/
│   └── 2_EDF_Viewer.py          [NEW] Streamlit 페이지 entry
├── src/
│   ├── edf_loader.py            [NEW] EDF 파싱 (pyedflib)
│   ├── hypnogram.py             [NEW] *-Hypnogram.edf annotation 파서
│   └── edf_plotting.py          [NEW] 시각화 (plotly_resampler)
├── tests/
│   ├── test_edf_loader.py       [NEW]
│   ├── test_hypnogram.py        [NEW]
│   ├── test_edf_plotting.py     [NEW]
│   └── fixtures/
│       ├── _make_synth.py       [NEW] 합성 EDF 생성기
│       ├── synth_psg.edf        [NEW] 30s × 3ch 합성 EDF
│       └── synth_hypnogram.edf  [NEW] 합성 hypnogram
└── requirements.txt              [MOD] +pyedflib, +plotly-resampler
```

기존 `app.py`, `src/loader.py`, `src/plotting.py`, `src/detector.py`, `src/evaluator.py`는 **수정 없음**.

### 4.2 의존 그래프

```
pages/2_EDF_Viewer.py
        │
        ▼
src/edf_plotting.py
        │
        ├─→ src/edf_loader.py
        └─→ src/hypnogram.py
```

`pages/`는 비즈니스 로직 0줄. `src/edf_*.py`는 Streamlit을 import하지 않음(테스트 용이).

### 4.3 모듈 책임

#### `src/edf_loader.py`

데이터 클래스:
- `ChannelInfo(label, sample_rate, n_samples, physical_dim, group)` — `group ∈ {EEG, EOG, EMG, Resp, SpO2, Other}`
- `EdfMeta(path, subject_id, start_datetime, duration_sec, channels: list[ChannelInfo])`

함수 (모두 순수 함수):
- `list_edf_files(root: Path) -> list[Path]` — `*-Hypnogram.edf` 제외
- `load_meta(path: Path) -> EdfMeta` — 헤더만 읽음
- `load_signal(path: Path, ch_idx: int) -> np.ndarray` — 단일 채널 lazy load
- `classify_channel(label: str) -> str` — 정규식 룰 기반 그룹 라벨링
- `pair_hypnogram(psg_path: Path) -> Path | None` — 같은 디렉토리에서 2단계 매칭: ①정확 stem 치환 `-PSG` → `-Hypnogram`, ②subject-night prefix(stem 첫 7자) + `*-Hypnogram.edf` glob

#### `src/hypnogram.py`

- `Stage` enum: `W / N1 / N2 / N3 / REM / MOVE / UNK`
- `HypnogramEpoch(start_sec: float, duration_sec: float, stage: Stage)`
- `load_hypnogram(path: Path) -> list[HypnogramEpoch]` — pyedflib `readAnnotations()` 파싱
- `to_stage_series(epochs, total_dur_sec) -> pd.DataFrame` — 1Hz 시각화용 시계열

`edf_loader.py`에 의존하지 않음 (annotation 파서가 독립적).

#### `src/edf_plotting.py`

- `make_freezoom_figure(meta, signals: dict[int, np.ndarray], hypno, channels: list[int]) -> Figure`
  - `FigureResampler` wrap → 자동 LTTB
  - 채널별 stacked subplot, `shared_xaxes=True`
  - hypnogram이 있으면 하단 색띠 trace 추가
- `make_epoch_figure(meta, signals, hypno, channels, epoch_idx: int, epoch_sec: float = 30) -> Figure`
  - 30s 윈도우 슬라이스 (다운샘플 불필요)
  - 미니맵: 야간 전체 hypnogram + 현재 위치 수직선
- `CHANNEL_GROUP_COLORS: dict[str, str]` — 그룹별 일관 색상

#### `pages/2_EDF_Viewer.py`

UI 위젯 + 캐시 wrapper만:
- 사이드바: 파일 selectbox → 채널 그룹 토글 5개 → 개별 채널 multiselect → 모드 라디오 → epoch 슬라이더(epoch 모드 시)
- 메인: 헤더 패널 → 차트 → 캡션
- `@st.cache_data`로 `load_meta`, `load_signal(path, ch_idx)`, `load_hypnogram` 감쌈

---

## 5. 데이터 흐름

```
[CAPA_Data/]
    │
    ├── list_edf_files() ─→ 파일 목록 (cache TTL 60s)
    │
[사용자 파일 선택]
    │
    ├── load_meta(path) ─→ EdfMeta
    │     ├─→ 헤더 패널 표시
    │     └─→ classify_channel() → 그룹 토글 UI 생성
    │
    ├── pair_hypnogram(path) ─→ Optional[Path]
    │     └─ load_hypnogram() ─→ list[HypnogramEpoch]
    │
[사용자 채널/모드 선택]
    │
    ├── 선택된 채널마다 load_signal(path, ch_idx) ─→ np.ndarray (lazy + cache)
    │
[모드 분기]
    ├── 자유 zoom  → make_freezoom_figure() → FigureResampler (zoom 시 자동 LTTB)
    └── 30s epoch → make_epoch_figure(epoch_idx)
    │
    ▼
st.plotly_chart(fig, use_container_width=True)
```

---

## 6. 캐시 전략 (`@st.cache_data`)

| 함수 | 키 | 크기 추정 | TTL / max_entries |
|---|---|---|---|
| `list_edf_files(root)` | root str | ~1 KB | TTL 60s |
| `load_meta(path)` | path str | ~5 KB | 영구 |
| `load_hypnogram(path)` | path str | ~50 KB | 영구 |
| `load_signal(path, ch_idx)` | (path, ch_idx) | 채널당 ~12 MB | `max_entries=10` (LRU) |
| Figure 함수 | — | — | **캐시 안 함** (FigureResampler stateful) |

**원칙**: 신호는 채널 단위 lazy load. 7채널 모두 켜도 ~80 MB로 Streamlit Cloud 한도 내. Figure는 매 렌더 새로 생성(CPU 비용 무시 가능).

---

## 7. 에러 처리와 엣지 케이스

| 상황 | 위치 | 대응 |
|---|---|---|
| `CAPA_Data/`에 EDF 0개 | `list_edf_files` | `st.info` + `st.stop()` |
| 손상된 EDF 헤더 | `load_meta` | try/except → `st.error(f"EDF 헤더를 읽을 수 없습니다: {e}")` |
| 채널 0개 선택 | UI | `st.warning("채널을 1개 이상 선택하세요.")` + 차트 placeholder |
| Hypnogram 페어링 실패 | `pair_hypnogram` | silent. `st.caption("ℹ️ Hypnogram 파일이 없어 sleep stage 오버레이가 비활성화되었습니다.")` |
| Hypnogram 파싱 실패 | `load_hypnogram` | `st.warning` 1회 + 오버레이 생략, 차트는 정상 |
| Hypnogram 길이 ≠ PSG duration | `load_hypnogram` | 짧은 쪽에 truncate + 안내 |
| sample_rate=0 인 채널 | `load_signal` | 해당 채널 skip + warning |
| 매우 큰 EDF (duration > 24h, ch > 50) | `load_signal` | 첫 6시간만 슬라이스 + warning |
| epoch_idx 범위 초과 | UI | 슬라이더 min/max로 차단 (불가) |
| 채널 그룹 분류 실패 | `classify_channel` | `Other`로 fallback (실패 아님) |

**검증 원칙**:
- EDF 형식 검증은 pyedflib에 위임. 자체 magic-byte 체크 없음.
- 위젯이 유효 옵션만 노출하므로 본문 추가 검증 생략.
- 내부 함수 호출은 신뢰 (boundary에서만 가드).

### Hypnogram 페어링 규칙

Sleep-EDF Cassette 표준은 PSG와 Hypnogram의 마지막 1자가 다름(예: `SC4001E0-PSG` ↔ `SC4001EC-Hypnogram`). 따라서 단순 stem 치환만으로는 매칭이 실패합니다. 2단계 규칙:

1. **정확 매칭**: `stem.replace('-PSG', '-Hypnogram') + '.edf'`가 같은 디렉토리에 존재하면 그것을 반환. (자체 데이터 / 표준에서 벗어난 명명에 강건)
2. **prefix 매칭**: 1단계 실패 시 subject-night prefix를 PSG stem의 첫 7자로 정의(`SC4001E0-PSG` → `SC4001E`)하고 같은 디렉토리에서 `<prefix>*-Hypnogram.edf` glob 후 첫 매치 반환. (Sleep-EDF Cassette `SC4001E0-PSG` ↔ `SC4001EC-Hypnogram` 케이스 처리)
3. 둘 다 실패 시 `None` (silent miss).

prefix 길이 7은 Sleep-EDF Cassette 명명 규칙(`SC4ssNeo`에서 첫 7자가 subject+night까지) 기준. 다른 명명 규칙 EDF에서는 1단계가 잡거나 None이 됩니다.

---

## 8. 테스트 전략

| 계층 | 테스트 범위 | 이유 |
|---|---|---|
| `edf_loader.py` | 단위 테스트 (모든 public 함수) | 정확성 critical |
| `hypnogram.py` | 단위 테스트 | 형식 코너케이스 많음 |
| `edf_plotting.py` | smoke test (Figure 생성 성공) | 렌더 정확성은 수동 확인 |
| `pages/2_EDF_Viewer.py` | 테스트 안 함 | UI e2e 비용 대비 수익 낮음, 비즈니스 로직 0줄 원칙 |

### 픽스처 — `tests/fixtures/synth_psg.edf`

`pyedflib.EdfWriter`로 합성한 30초 × 3채널 EDF (~10 KB, git commit). 생성 스크립트 `_make_synth.py`는 한 번만 실행. 채널:
- ch0: 100 Hz EEG 모사 (sin 10Hz + 노이즈) → 그룹 EEG
- ch1: 100 Hz EOG 모사 (sin 1Hz) → 그룹 EOG
- ch2: 1 Hz SpO2 모사 (95–99 random walk) → 그룹 SpO2

같이 합성된 `synth_hypnogram.edf`는 `[(0, 30, "W")]` 한 epoch.

### 주요 테스트 케이스

`test_edf_loader.py`:
- `list_edf_files`가 `*-Hypnogram.edf`를 제외
- `load_meta`가 채널 수/길이/그룹 라벨링 정확
- `load_signal` shape이 meta와 일치
- `classify_channel`이 모르는 라벨에서 `Other` fallback
- `pair_hypnogram`이 정확 stem 치환 케이스 매칭 (`a-PSG.edf` ↔ `a-Hypnogram.edf`)
- `pair_hypnogram`이 Sleep-EDF Cassette prefix 케이스 매칭 (`SC4001E0-PSG.edf` ↔ `SC4001EC-Hypnogram.edf`)
- `pair_hypnogram`이 hypnogram 없으면 `None`
- `load_meta`가 손상 파일에서 OSError/ValueError raise

`test_hypnogram.py`:
- `load_hypnogram`이 epoch 정확히 파싱
- `to_stage_series`가 total_dur로 truncate
- `Stage` enum이 AASM 5단계(W/N1/N2/N3/REM) 포함

`test_edf_plotting.py` (smoke):
- `make_freezoom_figure`가 trace 생성 (data non-empty)
- `make_epoch_figure(epoch_idx=0)`가 trace 생성

### 수동 검증 체크리스트 (PR 시)

- [ ] `streamlit run app.py` → 사이드바에서 "EDF Viewer" 페이지 진입
- [ ] `SC4001E0-PSG.edf` 로드 → 헤더 패널에 채널 표시
- [ ] EEG Fpz-Cz 선택 → 신호 차트 렌더, 8h 분량 부드러운 zoom
- [ ] Hypnogram 색띠가 차트 하단에 표시
- [ ] 30s epoch 모드 토글 → 슬라이더로 이동
- [ ] 1번 페이지(CAPA Analyzer)가 기존 동작 그대로

---

## 9. 의존성 추가

`requirements.txt`:

```
+ pyedflib>=0.1.36
+ plotly-resampler>=0.10
```

- `pyedflib`: pure Python wheel, manylinux 지원, Streamlit Cloud OK
- `plotly-resampler`: `dash`도 같이 설치되지만 Plotly Figure 모드만 사용하므로 무해

---

## 10. UI 라벨 (한국어 + 영어 병기)

기존 1번 페이지 패턴(`Precision (정밀도)`) 유지:
- `채널 그룹 (Channel Groups)`
- `에포크 (Epoch)`
- `자유 줌 모드 (Free Zoom)`
- `EDF 헤더 (Header)`
- `수면 단계 (Sleep Stage)`

---

## 11. 향후(v2 후보)

- 신호 필터 (`scipy.signal` band-pass / notch)
- 선택 구간 DataFrame export
- 사용자 EDF 업로드 (`st.file_uploader`)
- 멀티 EDF 좌우 비교 split view
- mne로 마이그레이션 (artefact 분석 시)

---

## 12. 학술/표준 참조

- **EDF/EDF+ 형식**: Kemp B, Olivan J. *Clin Neurophysiol* 2003; 114: 1755–1761
- **Sleep-EDF Database**: PhysioNet, https://physionet.org/content/sleep-edfx/
- **AASM Sleep Staging**: Berry RB et al. *J Clin Sleep Med* 2012; 8(5): 597–619
- **LTTB 알고리즘**: Steinarsson S. "Downsampling Time Series for Visual Representation" (MSc thesis, 2013)
- **plotly-resampler**: Van Der Donckt et al. *SoftwareX* 2022; 19: 101231
