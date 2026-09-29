# 항적수사대 · 2주차

공간 특징을 검산하고, 보지 못한 선박의 `fishing_code`를 분류하는 LightGBM 기준모형을 만든다.
원본→항적 확인→특징→EDA→선박 분할→학습→평가를 네 명 모두 실행한다.
이상탐지 구현은 4주차 범위이므로 포함하지 않았다.

현재 전체 실행 결과: **3,887,065행·17개 특징, 선박 교집합 0, Accuracy 37.03%, Macro-F1 0.3378**.
다수 클래스 기준 Accuracy 43.44%보다 낮다. 현재 한계를 포함한 해석은 [2주차 분석 기록](WEEK2_REPORT.md)에 있다.

## 실행

Python 3.11, 프로젝트 폴더에서 PowerShell로 실행한다. 원본 CSV 폴더는 이동할 필요가 없다.
현재 `.venv`에는 실행에 필요한 패키지가 설치되어 있다. 다른 PC에서는 다음과 같이 준비한다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

전원이 아래 순서로 실행한다. 전체 원본이 약 5.86GB이므로 특징 생성은 수 분 이상 걸린다.
동시에 같은 출력 폴더에 실행하지 말고 각자 로컬 복사본에서 실행한다.

```powershell
# STEP 1~2: 전체 표 점검 + 실제 항적 3개
.\.venv\Scripts\python.exe -X utf8 01_data_check.py

# STEP 3: 간단한 정답이 있는 경계 사례와 실제 항적 5개 검산
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -X utf8 02_feature_analysis.py --examples-only

# STEP 4: 원본 전체의 feature_df.csv 생성 + EDA
.\.venv\Scripts\python.exe -X utf8 -u 02_feature_analysis.py

# STEP 5~6: 선박 분할 + 학습 + 평가 + 오분류 확인
.\.venv\Scripts\python.exe -X utf8 -u 03_baseline.py
```

특징을 다시 계산하지 않고 EDA만 반복하려면 `02_feature_analysis.py --eda-only`를 실행한다.
각 스크립트는 담당 산출물을 다시 생성한다. 원본 파일은 수정하지 않는다.
`feature_df.csv`는 생성이 끝난 뒤 사용한다. 완료 여부는 `feature_audit.json`의 월별 행 수와 실행 종료 코드로 확인한다.

## 먼저 읽은 자료와 실제 구조

`2026_참가신청서_항적수사대 (1).pdf` 3~5쪽에서 계획과 누출 조건을 확인했다.
HWP도 같은 이름으로 제공되어 있으며 본문 확인에는 PDF를 사용했다.
기존 Python 코드, README, 별도 샘플, 기존 결과물은 없었다. Git 저장소도 아직 초기화되어 있지 않다.

| 단계 | 신청서 계획 | 이번 작업 반영 |
|---|---|---|
| 문제·목표 | 신규 선박 업종 분류와 정상 패턴 이탈 조기 탐지 | 이번 주는 업종 분류까지 |
| 1주차 | 정의·시각화·결측·단절·label 점검, 데이터 사전·분할 기준 | 기존 결과가 없어 원본부터 재확인 |
| 2주차 | 공간 특징·분류 기준모형 | 특징 코드, 전체 table, LightGBM 평가 |
| 3주차 | 선박 그룹 교차검증·시간대 집계·시계열 비교 | 다음 실험으로 기록 |
| 4~6주차 | 정상 기준·이상점수, 시간순 검증·오경보, 제출 | 구현하지 않음 |
| 검증·누출 | rfid_id 입력 제외, 신규 선박 검증, 미래 이력 사용 금지 | 고정 GroupShuffleSplit 1회, 과거 집계도 아직 없음 |

현재 원본은 2026-01-01~06-30, 181일의 CSV 6개, 총 **3,887,069행**이다.
메타데이터가 손상된 6월 2행을 제외한 시점에서 7,372척이며 7,196척은 여러 행에 나타난다.
한 선박의 label은 모두 동일하다. label은 연안복합·자망·채낚기연승낚시·통발의 4종이다.
이 label을 조업/비조업 순간 상태나 사고 여부로 해석하지 않는다.

| 컬럼 | 확인 결과 | 용도 |
|---|---|---|
| `part_dt` | YYYYMMDD, 날짜 181종 | 원본 연결·분석 단위 |
| `rfid_id` | 반복되는 선박 식별자 | 그룹 분할 전용 |
| `rfid_time` | 정상 행 전체에서 정수 0~23 | 시간대 구분; 시간대·생성 규칙은 미확인 |
| `geom` | 2차원 WKT 항적, MultiLineString 표본 | 공간 특징 |
| `fishing_code` | 4개 업종명 | 정답 |
| `Unnamed:*` | 4월 5열, 6월 68열 | 빈 열 검증; 값이 있는 손상 행은 별도 보존 |

기본 dtype, 각 파일 shape, head, 모든 컬럼의 결측 수는 `outputs/data_check.txt`에 실제 출력으로 저장했다.
6월 손상 행 때문에 자동 추론 dtype이 정수·실수·문자열로 혼합된다. 숫자 변환 실패를 검사한 뒤 정상 행의 ID·날짜·시간을 정수로 통일한다.

정상 메타데이터에서 **선박×날짜×시간대 중복은 0건**이다. 따라서 원본 한 행을 분석 단위로 유지한다.
선박×날짜로 합치면 서로 다른 시간대를 섞으므로 이번 주에는 집계하지 않는다.
`source_file_id`(1~6월 순서)와 `source_row`(헤더 제외, 0부터 시작)로 원본을 추적한다.

신청서와 실제 데이터의 차이·미확인 사항:

- 신청서가 지적한 좌표별 시각 부재는 실제에서도 확인된다. `rfid_time`은 행 단위 값이며 각 점의 관측 간격·시작/끝 시각이 아니다.
- 추가 열은 완전히 비어 있을 것으로 단정할 수 없다. 6월에는 좌표가 다른 필드로 밀린 행과 label이 누락된 인접 행이 있다. 임의 복원하지 않는다.
- 2월 `geom` 결측은 2행이다. 최종 제외 목록은 `excluded_rows.csv`에 남긴다.
- CRS 메타데이터는 없다. 좌표 범위가 한국 주변 경도·위도에 부합하므로 **경도, 위도 순서의 도 단위**로 판단했다. EPSG:4326이 확인됐다고 주장하지 않는다. 전체 범위는 `feature_audit.json`에 저장한다.
- 사고 Excel은 167행·18열, 결측 0건이며 선박 ID가 없다. 파일명의 “테스트 데이터”를 신규 선박 분류 정답 데이터로 사용하지 않는다. 사고 선박과 항적의 직접 연결은 확인되지 않았다.

## 특징 정의와 계산 원칙

거리에는 반지름 6,371,008.8m의 Haversine을 사용한다. 위경도에 단순 유클리드 거리를 적용하지 않는다.
투영 CRS를 확정할 근거가 없는 baseline에서 미터 단위를 확보하는 간단한 구면 근사다.
타원체 측지거리와 차이가 있을 수 있으며, 원본 CRS와 선분 생성 규칙은 제공기관에 확인할 항목이다.

| 특징 | 정의·읽는 법 |
|---|---|
| `total_distance_m` | 같은 선분의 인접 점 거리 합. 관측되지 않은 이동은 포함하지 않음 |
| `segment_chord_sum_m` | 각 선분 시작·끝 거리 합. 행 전체의 첫·마지막 점을 잇지 않음 |
| `straightness` | 위 두 값의 비율, 0~1. 이동거리 0이면 NaN |
| `mean/max_direction_change_deg` | 같은 선분에서 유효한 이동 방향 간 절대 변화, 0~180도 |
| `turn_count_45`, `turn_rate_45` | 45도 이상 변화 횟수, 유효 방향 변화 중 비율. 시간당 빈도 아님 |
| `total_turn_deg` | 절대 방향 변화 합. 원형 선회 횟수나 회전 방향 아님 |
| `spatial_spread_m` | 모든 관측점의 평균 위치로부터 구면거리의 RMS |
| `bbox_width_m`, `bbox_height_m` | bbox 중간 위도에서 폭, 중간 경도에서 높이 |
| `num_segments`, `num_points` | 원본 선분 수·전체 점 수. 중복점 포함 |
| `duplicate_ratio` | 1 − 행 안의 고유 좌표 수 / 전체 점 수 |
| `zero_step_ratio` | 같은 선분 인접 좌표쌍 중 정확히 같은 좌표 비율 |
| `num_direction_changes` | 방향 변화를 계산할 수 있었던 횟수 |
| `max_step_m` | 같은 선분 내 최대 인접점 거리. 큰 값은 품질 점검 대상 |

총 17개 특징이다. 공간적 분산과 bbox는 모든 관측점을 대상으로 하되 선분 사이 이동을 가정하지 않는다.
길이 0의 이동은 방향이 없어 제외한다. 유효 이동이 2개 미만이면 평균·최대 방향 변화와 비율이 NaN이다.
원본의 반복 좌표는 점 수·분산·중복 지표에 보존한다. GPS 흔들림과 긴 점 간 간격을 임의의 임계값으로 제거하지 않는다.

속도·가속도·체류시간: **현재 데이터에서는 사용하지 않음**. 1시간을 모든 항적의 실제 관측 소요시간으로 간주하지 않는다.
시간대 간 변화·과거 집계는 3주차 후보이며 이번 특징에 없다. 위 공간·품질 특징들은 이후 이상점수 분석에 재사용할 수 있다.

`src/features.py`의 작은 함수들이 각 의미를 담당한다. 대용량 계산부의 `segment_rows`는 선분이 속한 원본 행,
`point_segments`는 점이 속한 선분이다. 이 인덱스의 경계를 확인해 다른 선분을 연결하지 않는다.
전체 388만 행은 chunk 단위의 NumPy 계산으로 처리하며 초보 팀원은 먼저 실제 5개 예제의 그림과 표를 비교한다.

## 검증을 읽는 법

그림은 `outputs/figures/`에 저장된다. `raw_tracks.png` → `feature_checks.png` →
`label_distribution.png` → `feature_distributions.png` → `confusion_matrix.png` →
`feature_importance.png` → `misclassified_tracks.png` 순서로 읽는다.

`feature_check_examples.csv`에는 그림에 사용한 원본 WKT와 계산값, 별도의 구면 코사인 법칙으로 합산한 거리도 있다.
직선적 사례의 직선성, 복잡한 사례의 방향 변화, 넓은 사례의 분산을 비교하고,
중복이 많은 거의 정지한 사례에서 높은 방향 변화가 나올 수 있음을 함께 확인한다.

LightGBM 입력은 `FEATURE_COLUMNS`에 명시된 숫자 17개뿐이다. 선박 ID·label·시간·날짜·파일 번호·행 번호·절대 좌표는 입력에서 빠진다.
NaN은 LightGBM이 처리하며 전체 데이터로 평균 대치·표준화·특징 선택을 하지 않는다.
전체 EDA는 설명용이다. 결과를 보고 validation에 맞춰 임계값이나 특징을 바꾸지 않았다.

GroupShuffleSplit(seed=42)는 **선박의 20%**를 validation에 배정한다. 행의 20%와는 다를 수 있다.
`set(train_vessel_ids) & set(valid_vessel_ids)`가 비어 있는지 assert로 확인한다.
분할 명단은 `vessel_split.csv`, 클래스별 행·선박 수는 `metrics/split_class_counts.csv`에 있다.
검증은 새로운 선박의 같은 기간 성능이며 미래 기간 예측 성능을 보장하지 않는다.

150개 boosting 반복, 학습률 .05, leaf 31, 학습 데이터 기준 balanced class weight를 사용한다.
튜닝·early stopping·validation 기반 선택은 없다. 다수 클래스만 예측하는 기준값도 함께 계산한다.
Accuracy는 맞춘 행의 비율, Macro-F1은 **클래스별 F1의 동일 가중 평균**이다.
Recall은 실제 그 업종의 행 중 맞춘 비율이고, confusion matrix의 행 정규화 대각선에서 읽는다.
행이 많은 선박이 평가에 더 많이 반영되므로 선박별 평가와 그룹 교차검증은 다음 주에 확인한다.
중요도는 분할 gain이며 인과관계나 독립적인 효과 크기가 아니다. 상관된 특징끼리 중요도가 나뉠 수 있다.
저장한 확률은 보정된 확률이 아니다.

## 네 명의 공통 실행과 교차검산

모두 같은 pipeline을 baseline까지 직접 실행한다. 역할은 이해 범위의 분리가 아니라 검산의 초점이다.

| 참여자 | 집중 확인 | 다른 사람에게 받는 확인 |
|---|---|---|
| 팀장 | 분석 단위·특징 정의·table·분할·결과 취합 | 팀원 2가 split 명단의 겹침을 별도로 확인 |
| 팀원 1 | 거리·직선성·방향 변화 함수 | 팀원 3이 그림과 실제 한 항적 수치를 대조 |
| 팀원 2 | 분산·bbox·점/선분 수·EDA | 팀원 1이 NaN·범위·중복값을 대조 |
| 팀원 3(초보) | 실제 항적→숫자→모델 입력, label 분포·오분류 | 팀장이 같은 table과 feature 목록 사용 여부 확인 |

팀원 3도 `03_baseline.py`를 직접 실행하고 잘못 분류된 실제 항적 2개를 설명한다.
전원은 혼동행렬을 함께 읽고 어느 두 업종을 왜 어려워하는지 가설을 적는다.
교차검산은 실제 팀원이 완료한 뒤 체크한다. 아래 체크는 자동 실행만으로 완료되었다고 간주하지 않는다.

STEP 1~2 완료 조건:

- [ ] 원본 한 행이 선박×날짜×시간대임을 설명할 수 있다.
- [ ] 실제 항적 하나의 선분·점·시작/끝을 직접 확인했다.
- [ ] 선분 사이 이동과 좌표별 시각을 만들어내면 안 되는 이유를 안다.

STEP 3~4 완료 조건:

- [ ] 다른 사람이 거리 하나와 선분·점 수 하나를 검산했다.
- [ ] NaN이 결함인지, 계산 불가능한 정의인지 구분했다.
- [ ] feature table 한 행과 모델에 들어가는 17개 숫자를 찾을 수 있다.

STEP 5~6 완료 조건:

- [ ] 다른 사람이 선박 교집합 0과 ID의 입력 제외를 확인했다.
- [ ] 각자 학습을 실행하고 Macro-F1과 클래스별 Recall을 읽었다.
- [ ] 전원이 혼동행렬·오분류를 보고 다음 실험 하나를 정했다.

## 산출물

| 파일 | 내용 |
|---|---|
| `outputs/feature_df.csv` | 재생성 가능한 전체 공간 특징 table |
| `outputs/data_check.txt`, `data_summary.json` | 실제 원본 출력·단위·결측 점검 |
| `outputs/invalid_source_rows.csv`, `excluded_rows.csv` | 원본 손상·geometry 제외 사유와 위치 |
| `outputs/feature_check.txt`, `feature_audit.json` | head·shape·info·describe·NaN·inf·범위 |
| `outputs/metrics/` | EDA 통계·높은 상관·큰 이동 간격·분할·평가·중요도 |
| `outputs/lightgbm_baseline.txt` | LightGBM 모델; 입력 순서·class 순서는 metrics의 JSON 참고 |
| `outputs/validation_predictions.csv` | 원본 위치를 포함한 validation 예측·확률 |
| `outputs/misclassified_examples.csv` | 실제 WKT를 포함한 오분류 사례 |
| `WEEK2_REPORT.md` | 실제 결과 해석과 3주차 확인 사항 |

API 동작 확인: [GroupShuffleSplit 공식 문서](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html),
[LightGBM 파라미터 공식 문서](https://lightgbm.readthedocs.io/en/latest/Parameters.html).
Git에는 코드·보고서·참가신청서(PDF/HWP)·집계 지표·그림·학습 모델을 포함한다. 원본 항적 CSV, 사고 Excel, 행 단위 파생 table 및 원본 행 미리보기는 `.gitignore`로 제외하며 로컬에서 별도로 준비한다.

## 차주 진행 계획

[3주차 성능 개선 계획](WEEK3_PLAN.md)에 baseline 비교 실험, 코드 수정 위치, 평가 원칙, 일정 및 결과 기록 양식을 정리했다. 해당 문서는 실험 전 계획이며 성능 개선 결과를 의미하지 않는다.
