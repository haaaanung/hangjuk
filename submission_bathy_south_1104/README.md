# 수심·남해 적용 제출 후보 (1104)

> **상태: 전체 기간의 동일 설정 5-fold 검증 전. 제출 채택은 보류합니다.**
> 이 폴더는 공유용 후보 보관본입니다. 엑셀 생성·형식 검산 완료는 성능 검증 완료를 뜻하지 않습니다.
> 기존 0.6052는 수심·퇴적물·어장·바람을 모두 사용하는 다른 구성의 개발 점수입니다. 이 후보의 점수로 인용하면 안 됩니다.

제출 양식으로 생성한 파일은 이 폴더의 **submission.xlsx 하나**입니다. 현재는 성능 검증 전 후보로 보관합니다. 테스트A는 신규 후보이며 테스트B의 셀 값은 기존 제출과 동일합니다. baseline_workbook.bin은 기존 통합 엑셀의 재현용 원본 바이트이며 제출하지 않습니다.

## 실행

Python 3.11 이상에서 다음을 실행합니다.

```powershell
python -m pip install -r requirements.txt
python predict.py --zip "C:\경로\문제 데이터_수정_1103_v2.zip"
```

포함된 모델과 수심 격자를 사용하며 재학습이나 외부 다운로드를 하지 않습니다. 공식 데이터 ZIP은 별도로 필요합니다.

## 고정한 방식

- 입력: 현재 항적 행의 34특징과 GEBCO_2026 수심·경사 14특징. 선박 ID 및 과거 항적 프로필은 모델 입력에 사용하지 않습니다.
- 모델: L127 설정(350 trees, 127 leaves, min_child_samples 50), 위치 2열과 기록 특성 5열의 상호작용 차단.
- 현재 행 중심이 126.5 ≤ 경도 < 129.3, 위도 < 35.3이면 수심 모델, 나머지는 기존 L127 모델의 확률을 사용합니다.
- 선박별 모든 유효 행의 확률을 평균해 최종 업종과 해당 확률을 출력합니다. 기존 제출의 활동 행 평균과 구분됩니다.
- 최종 학습 범위는 개발 Fold 1~5입니다. Fold 0은 사용하지 않았습니다.
- 남해 규칙은 노출된 개발 자료를 본 뒤 선택했습니다. 시간 분리 검증에서 개선 방향이 재현됐으나 실제 제출 점수의 개선은 미확인입니다.

## 파일

- submission.xlsx: 제출할 통합 엑셀
- model/: 기존 L127 및 수심 모델과 학습 설정
- grid/: GEBCO_2026 대한민국 주변 부분 격자의 전처리 배열 및 메타데이터
- predict.py, base_predict.py, current_features.py, bathy_features.py: 추론 코드
- train.py: 원래 프로젝트의 Fold 1~5 캐시로 학습을 재현하는 코드
- VERIFICATION.json: ID 순서, 확률, 테스트B 보존, 변경 수 등 검산 결과
- comparison.json: 기존 제출 및 L127 전체 평균과의 예측 비교

GEBCO 출처: https://www.gebco.net/data-products/gridded-bathymetry-data
수심은 GEBCO_2026 원격 격자의 값이며, 정확한 실제 해저 측량이나 항해 용도로 보장되지 않습니다. 원본 제공자료의 이용 조건은 동봉한 GEBCO 문서를 참고하세요.

GEBCO 출처 표기: GEBCO Bathymetric Compilation Group 2026 (2026). The GEBCO_2026 Grid. doi:10.5285/4f68d5c7-45eb-f999-e063-7086abc036fa
