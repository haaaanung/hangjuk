# 직접 학습하기

> 최종 모델은 `outputs/experiments/final_v1/model.txt`로 확정했습니다. 최신 결론과 실측 결과는 [최종 보고서](MODEL_IMPROVEMENT_REPORT.md)를 참고하세요. 채택된 설정만 재학습하려면 `10_train_improved.py --run-id control_retrain --config configs/final_training.json`을 사용합니다. 아래는 이전의 여러 후보 비교 실험 안내입니다.

## 1. 실행 명령

프로젝트 폴더에서 PowerShell을 열고 실행하세요.

```powershell
.\.venv\Scripts\python.exe -X utf8 -u 10_train_improved.py --run-id improved_v1
```

**특징 준비 → 선박 단위 교차검증 → 후보 선택 → 최종 학습 → 결과 보고서** 순서로 진행합니다.
이 후보 실험은 `improved_v1`에서 이미 완료됐습니다. 아래 명령을 새로운 설정으로 실행할 때는 다른 run-id를 사용하세요.

최종 비교와 후보 채택 여부는 최종 보고서에만 기록합니다.

필요한 파일은 현재 프로젝트에 있습니다.

- `outputs/feature_df.csv`: 기존 관측별 특징
- `outputs/experiments/context_v1/vessel_split.csv`: 기존 학습/검증 선박 명단
- `outputs/experiments/context_v1/cv_assignments.csv`: 기존 3-fold 명단
- 원본 CSV 폴더: 특징 캐시를 다시 만들 때 사용

다른 PC에서는 먼저 환경을 준비하세요. 새 학습기는 기존 `requirements.txt`의 패키지만 사용합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 2. 이번에 바뀐 점

저장된 결과에서는 통발 Recall이 약 **0.313**, F1이 **0.359**로 낮았습니다. 깊은 트리와 강한 규제만 적용한 기존 후보도 내부 검증에서 기존 모델을 넘지 못했습니다. 이번 실행은 다음을 함께 비교합니다.

| 변경 | 목적 |
|---|---|
| 현재 움직임과 과거 평균의 차이, 최근 7일/30일 변화, 과거 활동·정지 위치에서의 거리 | 평소 행동과 현재 행동의 관계를 표현 |
| 과거 프로필 중심 모델 | 개별 시간대의 항적 형태에 덜 의존하는 후보 추가 |
| 클래스·선박별 가중치 | 소수 업종을 고려하고 관측이 많은 선박의 영향 완화 |
| 선박당 학습 상한 기본 128→256행 | 각 선박의 다양한 활동을 더 많이 학습 |
| 학습 선박 내부의 별도 선박으로 조기 종료 | 반복 수를 자동 선택하고 과도한 학습 억제 |
| 모델 확률 평균 앙상블 | 서로 다른 후보의 오분류를 보완하는지 확인 |
| 이전 날짜 예측의 누적 평균 | 업종이 일정한 선박의 시간대별 예측 흔들림 완화 |

**예측 누적은 정답을 사용하지 않습니다.** 같은 선박의 엄격히 이전 날짜 예측만 사용하며, 하루에 관측이 많아도 하루당 한 표로 평균합니다. 현재 날짜의 다른 관측과 미래 날짜는 제외합니다. 첫날에는 현재 예측을 그대로 사용합니다.

누적 반영 비율은 `설정 비율 × 이전 관측일 수 / (이전 관측일 수 + 3)`입니다. 이력이 쌓일수록 서서히 반영됩니다. 기본 후보는 최대 반영 비율 0%, 50%, 80%이며, 누적이 해롭다면 0%도 선택할 수 있습니다.

## 3. 결과 확인

가장 먼저 **`outputs/experiments/improved_v1/RESULTS.md`**를 읽으세요.

| 파일 | 내용 |
|---|---|
| `RESULTS.md` | 정확도·Macro-F1 비교, 업종별 Precision/Recall/F1, 해석 |
| `cv_summary.csv` | 후보 조합별 내부 검증 평균·표준편차, 선택 가능 여부 |
| `selection.json` | 선택 모델과 누적 비율 |
| `metrics.json` | 최종 검증의 대조 모델 / 선택 모델 누적 전 / 누적 후 점수 |
| `selected_confusion_matrix.csv` | 어떤 업종끼리 혼동하는지 |
| `history_slices.csv` | 첫날 / 이전 관측일 1~6일 / 7일 이상의 성능 |
| `model_*.txt`, `model_metadata.json` | 모델 및 재사용에 필요한 특징·클래스 순서·앙상블 비중 |

후보 선택은 **내부 CV 평균 Accuracy와 Macro-F1이 모두 이번 실행의 대조 모델 이상**인 조합으로 제한합니다. 그중 두 지표 평균이 가장 높은 조합을 선택합니다. 동점이면 모델 수와 누적 비율이 작은 쪽을 우선합니다.

대조 모델은 기존 `behavior_sqrt` 구조를 이번 설정의 학습 표본 수로 재학습합니다. 기존에 저장된 60.55%/0.5993과 동일한 실험은 아닙니다. 최종 검증에서도 두 지표가 모두 개선됐는지 `RESULTS.md`에서 별도로 확인합니다. 개선되지 않아도 검증 결과에 맞춰 후보를 자동 재선택하지 않습니다.

## 4. 설정 바꾸기

설정 파일: **`configs/improved_training.json`**

| 항목 | 기본값 | 조정 방법 |
|---|---:|---|
| `threads` | 4 | CPU 사용량. 8로 늘리면 빨라질 수 있지만 데이터 복사 메모리는 줄지 않음 |
| `cap_per_vessel` | 256 | CV와 최종 학습에서 동일하게 적용. 메모리가 부족하면 128, 더 넓게 학습하려면 512. 0은 전체 행 사용 |
| `early_stopping_rounds` | 100 | 내부 검증 손실이 개선되지 않을 때 기다릴 반복 수 |
| `history_strengths` | `[0.0, 0.5, 0.8]` | 누적 비율 후보. 누적 미사용 비교를 위해 0 필수 |
| `history_warmup_days` | 3 | 누적 반영이 커지는 속도. 클수록 더 천천히 반영 |
| 후보별 `rounds` | 350 또는 1600 | 대조 모델은 고정 반복. 나머지는 조기 종료 전 최대 반복 수 |
| 후보별 `class_weight_power` | 0.35~0.75 | 클수록 소수 클래스 비중 증가. Accuracy와 Macro-F1을 함께 확인 |
| 후보별 `vessel_weight_power` | 0~0.5 | 클수록 반복 관측이 많은 선박의 영향 감소 |

기본 후보는 대조 모델·확장 특징 모델·과거 프로필 모델 3개입니다. CV 중 두 개선 후보는 내부 조기 종료 학습 후 해당 fold를 다시 학습하므로 **기본 설정에서 CV 학습은 총 15회**, 최종 학습은 선택 결과에 따라 1~3회입니다. 특징 캐시 크기만 약 2.5GB이며 학습용 복사본 등 추가 메모리가 필요합니다. 정확한 소요 시간은 아직 측정하지 않았습니다.

설정을 바꿀 때는 **새로운 run-id**를 사용하세요.

```powershell
# CPU 스레드 수를 명령에서 지정
.\.venv\Scripts\python.exe -X utf8 -u 10_train_improved.py --run-id improved_threads8 --threads 8

# 설정 수정 후 별도 실험
.\.venv\Scripts\python.exe -X utf8 -u 10_train_improved.py --run-id improved_v2 --config configs/improved_training.json
```

## 5. 중단·재개 및 단계별 실행

중단됐다면 **동일한 설정과 동일한 명령**으로 다시 실행하세요. 완료된 후보×fold의 예측과 완료된 최종 모델은 재사용합니다. 진행 중이던 단일 학습은 처음부터 다시 수행합니다. 같은 폴더에 두 작업을 동시에 실행하지 마세요.

```powershell
# 설정과 필수 파일 존재 여부만 확인. 데이터 전체 점검이나 학습은 하지 않음
.\.venv\Scripts\python.exe -X utf8 10_train_improved.py --check-config

# 특징 캐시만 준비
.\.venv\Scripts\python.exe -X utf8 -u 10_train_improved.py --stage prepare

# 교차검증과 후보 선택까지만
.\.venv\Scripts\python.exe -X utf8 -u 10_train_improved.py --run-id improved_v1 --stage cv

# 선택된 모델의 최종 학습과 검증
.\.venv\Scripts\python.exe -X utf8 -u 10_train_improved.py --run-id improved_v1 --stage final
```

설정·코드·입력 데이터·분할·라이브러리 버전이 달라지면 기존 실행을 이어 쓰지 못하도록 검사합니다. 실험 결과를 삭제할 필요 없이 새 이름으로 실행하면 됩니다.

## 6. 평가와 재사용 원칙

- 학습/검증과 내부 CV 모두 선박 단위로 분리합니다. 선박 ID는 분할·이력 집계에만 사용합니다.
- 조기 종료용 선박도 해당 CV 학습 선박에서만 떼어냅니다. 외부 fold나 최종 검증 정답으로 반복 수를 고르지 않습니다.
- 최종 반복 수는 해당 후보의 3개 fold 반복 수 중앙값입니다. CV·최종 학습에서 동일한 선박별 표본 상한을 적용합니다.
- 정확도와 F1은 모든 검증 관측을 대상으로 계산합니다. 선박별 동일 가중 점수도 `metrics.json`에 저장합니다.
- CV 점수는 후보 선택에 사용한 점수입니다. 최종 검증도 이미 확인한 같은 기간의 개발용 선박 분할이며, 미래 기간에 대한 독립 평가가 아닙니다.
- 모델 파일 재사용 시 `src.training_features.feature_batch`로 모델별 입력을 만들고, `model_metadata.json`의 비중으로 확률을 평균합니다. 이후 `src.prediction_history`의 두 함수로 누적 규칙을 적용합니다. **행을 임의로 나눠 누적하면 이전 이력이 사라지므로 선박의 필요한 과거 예측도 함께 제공해야 합니다.**

## 기존 스크립트

모델 변경 실험은 `13_compare_architectures.py`에서 CatBoost와 ExtraTrees를 비교했습니다. 실측 결과와 최종 채택 여부는 [최종 보고서](MODEL_IMPROVEMENT_REPORT.md)를 따릅니다. 이 실험의 추가 패키지는 `requirements-models.txt`로 설치하며, 일반 LightGBM 재현에는 필요하지 않습니다.

| 파일 | 용도 |
|---|---|
| `04_improve.py` | 위치·과거 이력 모델 |
| `05_diagnose_features.py` | 특징·과적합 원인 진단 |
| `06_behavior_model.py` | 기존 행동 특징 모델 재현 |
| `07_spatial_model.py` | 지역별 업종 사전정보 실험 |
| `08_regularized_model.py` | 강한 규제 및 앙상블 실험 |
| `09_compare_models.py` | 기존 완료 실험 비교 |
| **`10_train_improved.py`** | **이번 개선 학습 실행** |

`09_compare_models.py --cv-runs`는 기존 실험 전용 형식입니다. 10번의 조합·누적 평가는 해당 실행의 `cv_summary.csv`를 사용하세요.
