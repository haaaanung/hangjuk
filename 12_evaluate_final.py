"""Verify the selected checkpoint on the fixed development vessel validation.

No retraining: the best completed checkpoint is reused. This is explicitly a
development-set selection, not an untouched-test performance claim.
"""

import argparse
import importlib.util
import json
import shutil
from pathlib import Path
from types import ModuleType
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.common import OUTPUT, ROOT
from src.context_features import sha256
from src.improved_training import save_json


def load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(Path(name).stem, ROOT / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_report(out: Path, metrics: dict[str, Any], bootstrap: dict[str, Any]) -> None:
    baseline, final = metrics["baseline"], metrics["selected"]
    failed = read_json(OUTPUT / "experiments/decision_v1/selection.json")
    class_report = pd.read_csv(out / "selected_classification_report.csv", index_col=0)
    class_lines = [
        f"| {label} | {class_report.loc[label, 'precision']:.4f} | "
        f"{class_report.loc[label, 'recall']:.4f} | {class_report.loc[label, 'f1-score']:.4f} |"
        for label in metrics["classes"]
    ]
    accuracy_interval = bootstrap["accuracy"]["paired_vessel_95pct_interval"]
    f1_interval = bootstrap["macro_f1"]["paired_vessel_95pct_interval"]
    report = f"""# 모델 개선 보고서

## 1. 초기 상태

- 상태: Phase 4 완료. 추가 실험 종료; 단일 모델 확정.
- 문제는 순간 조업/사고 탐지가 아닌 **선박 업종 4종 분류**다.
- 기존 `behavior_v2`: Accuracy {baseline['accuracy']:.6f}, Macro-F1 {baseline['macro_f1']:.6f}.
- 총 3,887,065행. 학습 선박 5,897척, 검증 선박 1,475척·707,233행. 기존 분할 유지; 선박 교집합 0, 선박×날짜×시간대 중복 0, 선박별 정답 일관성 확인.
- 학습 클래스 행 수: 연안복합 1,375,197 / 자망 1,156,988 / 채낚기연승낚시 141,619 / 통발 506,028. 소수 업종과 선박별 관측량의 불균형이 있다.
- 기존 통발 Recall 0.313/F1 0.359. 통발이 연안복합·자망으로, 연안복합과 자망이 서로 혼동되는 패턴이 핵심 병목이다.
- 기존 context 모델의 표본 학습/검증 Macro-F1은 0.9485/0.5794였다. 과적합 근거이며 최종 모델의 학습 점수로 해석하지 않는다.
- 좌표별 시각·측정 속력·침로가 없어 시간대 번호로 속도/가속도를 만들지 않았다. 원본은 AIS 필드가 완비된 자료가 아닌 VPASS 시간대별 geometry다.

## 2. 주요 개선 사항

- **이미 학습된 단일 LightGBM 재사용**: 전체 반복 관측을 쓰던 기존 최종 학습과 달리 선박당 최대 256행으로 제한한 체크포인트를 채택했다. CV와 최종 학습의 표본 상한도 일치한다. 표본 상한 외 차이를 완전히 분리한 인과 실험은 아니다.
- 과거 행동 특징 163개와 클래스 역빈도 제곱근 가중치는 유지한다. 과거 특징은 해당 선박의 엄격히 이전 날짜만 사용하고 ID는 모델 입력에서 제외한다.
- 별도 실험에서 기존 OOF로 클래스 판정 비중을 학습했다. 각 fold 안에서 선박 절반은 비중 학습, 나머지 절반은 평가에 사용했다. 기본 모델은 해당 fold 전체를 보지 않았으므로 평가 선박 정답의 간접 유입도 차단했다.
- 판정 비중 후보는 분리 평가에서 두 지표가 하락해 **폐기**했다. 이 후보의 최종 검증은 실행하지 않았다. 프로필 앙상블·확장 특징·강한 규제도 최종 모델에 추가하지 않았다.
- 이번 작업에서 트리 재학습은 0회, GPU 사용은 0회다. 기존 완료 학습을 반복하지 않고 판정 비중 학습, 누수 검사, 최종 전체 추론 및 선박 bootstrap만 수행했다.

## 3. 성능 비교

동일한 개발 검증 선박·707,233행으로 비교했다. 기존 기준 예측과 키·선박 ID·정답을 일치시켰고 두 저장 모델의 지표를 재현했다.

| 변경 사항 | 개선 목적 | Accuracy | Macro-F1 | 결과 |
|---|---|---:|---:|---|
| 기존 행동 특징 모델 | 비교 기준 | {baseline['accuracy']:.6f} | {baseline['macro_f1']:.6f} | 비교용 유지 |
| 선박당 최대 256행, 단일 LightGBM | 반복 관측 영향·과적합 완화 | {final['accuracy']:.6f} | {final['macro_f1']:.6f} | **최종 채택** |
| 프로필 앙상블+이전 날짜 누적 | 예측 안정화 | 0.612921 | 0.606969 | 최종 모델에서 제외 |

새 판정 비중 실험은 별도 선박으로 나눈 내부 평가이므로 위 표와 직접 비교하지 않는다.

| 변경 사항 | 개선 목적 | Accuracy | Macro-F1 | 결과 |
|---|---|---:|---:|---|
| 비중 조정 전 단일 모델 | 내부 비교 기준 | {failed['cv']['control']['accuracy']:.6f} | {failed['cv']['control']['macro_f1']:.6f} | 유지 |
| 클래스 판정 비중 학습+누적 | 클래스 혼동 완화 | {failed['cv']['adjusted']['accuracy']:.6f} | {failed['cv']['adjusted']['macro_f1']:.6f} | 폐기 |

최종 모델의 업종별 결과:

| 업종 | Precision | Recall | F1 |
|---|---:|---:|---:|
{chr(10).join(class_lines)}

- Accuracy 개선: **{(final['accuracy'] - baseline['accuracy']) * 100:+.3f}%p**. 선박 단위 paired bootstrap 95% 구간: [{accuracy_interval[0] * 100:+.3f}, {accuracy_interval[1] * 100:+.3f}]%p.
- Macro-F1 개선: **{final['macro_f1'] - baseline['macro_f1']:+.6f}**. 95% 구간: [{f1_interval[0]:+.6f}, {f1_interval[1]:+.6f}].
- 선박별 동일 가중 Accuracy/Macro-F1: {final['vessel_equal_accuracy']:.6f}/{final['vessel_equal_macro_f1']:.6f}. 행 단위 점수와 다르다.
- Bootstrap은 검증 선박 1,475척을 2,000회 재표집했다. 개발 검증의 조건부 불확실성이며 모델 선택 편향이나 미래 분포 변화는 포함하지 않는다.

## 4. 최종 모델

- 모델: `outputs/experiments/{out.name}/model.txt`.
- 입력·클래스 순서·출처: 같은 폴더의 `model_metadata.json`.
- 구조: 다중 클래스 LightGBM, 특징 163개, 350회, leaf 31, 학습률 0.05, 최소 leaf 표본 200, L2=5, seed=42, CPU 4 threads.
- 학습 표본: 5,897척에서 선박당 최대 256행, 총 1,044,833행. 클래스 가중치 지수 0.5; 별도 선박 가중치·예측 누적·앙상블 없음.
- 기존 학습 체크포인트: `outputs/experiments/improved_v1/model_control.txt`. 복사한 최종 모델의 SHA-256 일치를 확인했다.
- 원시 검증 예측·혼동행렬·클래스별 지표·paired bootstrap은 최종 폴더에 저장했다. 평가 재현: `python -X utf8 -u 12_evaluate_final.py`.

## 5. 결론

**Accuracy {final['accuracy']:.6f}, Macro-F1 {final['macro_f1']:.6f}; 두 지표 0.90 목표 미달.** 개선 폭은 작고, bootstrap 구간에 0이 포함되면 통계적으로 확정적인 개선으로 주장할 수 없다.

최종 모델은 이미 반복 확인한 **개발용 validation 성능을 기준으로 선택**했다. 독립 미사용 테스트 세트가 없으므로 독립 최종 테스트 성능이나 미래 기간 성능을 주장하지 않는다. 기존 전체 기간 모델로 임의의 미래 구간을 잘라 테스트라고 부르지도 않았다.

추가한 판정 학습의 일반화가 실패했고, 기존 깊은 트리·규제·확장 특징 실험도 뚜렷한 이득이 없었다. 비용 대비 기대효과가 낮아 종료했다. 이 결과가 현재 데이터에서 0.90 달성이 수학적으로 불가능함을 증명하는 것은 아니다.

다음 우선순위는 더 큰 모델보다 **미사용 신규 선박·후속 기간의 독립 평가 자료**, 신뢰할 수 있는 점별 시각/속력/침로, 선종·어구·등록항 등 실제 배포에서도 얻을 수 있는 메타데이터, 통발/연안복합/자망의 label 정의 점검이다.
"""
    (ROOT / "MODEL_IMPROVEMENT_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="final_v1")
    args = parser.parse_args()
    if not args.run_id.isascii() or not args.run_id.replace("_", "").replace("-", "").isalnum():
        parser.error("Use a simple ASCII run-id")
    out = OUTPUT / "experiments" / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    source = OUTPUT / "experiments/improved_v1"
    previous = OUTPUT / "experiments/behavior_v2"
    model_path = source / "model_control.txt"
    stamp = {
        "model_sha256": sha256(model_path),
        "source_config_sha256": sha256(source / "config.json"),
        "baseline_predictions_sha256": sha256(previous / "validation_predictions.csv"),
        "runner_sha256": sha256(Path(__file__)),
    }
    if (out / "config.json").exists():
        if read_json(out / "config.json") != stamp:
            raise ValueError("Inputs/code changed; use a new run-id")
        if (out / "metrics.json").exists():
            write_report(out, read_json(out / "metrics.json"), read_json(out / "paired_vessel_bootstrap.json"))
            print(json.dumps(read_json(out / "metrics.json"), ensure_ascii=False))
            return
    elif any(out.iterdir()):
        raise ValueError("Output folder is not empty")
    else:
        save_json(out / "config.json", stamp)
    inputs = load_script("11_refine_decisions.py")
    evaluate = load_script("04_improve.py")
    compare = load_script("09_compare_models.py")
    print("입력 해시·선박 분할·관측 중복 검사", flush=True)
    df, y, folds, config = inputs.load_inputs(source)
    rows = np.flatnonzero(folds == 0)
    manifest = read_json(OUTPUT / "cache/behavior_v2/features.json")
    if manifest["signature"] != config["signature"] or manifest["columns"] != config["base_features"]:
        raise ValueError("Feature cache differs")
    matrix = np.load(OUTPUT / "cache/behavior_v2/features.npy", mmap_mode="r")
    model = lgb.Booster(model_str=model_path.read_text(encoding="utf-8"))
    columns, labels = config["base_features"], config["classes"]
    if model.feature_name() != columns or matrix.shape != (len(df), len(columns)):
        raise ValueError("Model feature order or cache shape differs")
    print(f"전체 검증 {len(rows):,}행 추론", flush=True)
    probability = np.concatenate([
        model.predict(matrix[rows[start:start + 50000]], num_threads=4)
        for start in range(0, len(rows), 50000)
    ])
    selected = evaluate.evaluate("selected", y[rows], probability, df.iloc[rows].rfid_id.to_numpy(), labels, out)
    expected = read_json(source / "metrics.json")["control"]
    if any(not np.isclose(selected[key], expected[key], rtol=0, atol=1e-10) for key in expected):
        raise ValueError("Control metrics reproduction failed")
    predictions = df.iloc[rows].copy()
    predictions["prediction"] = np.asarray(labels)[probability.argmax(axis=1)]
    for index, label in enumerate(labels):
        predictions[f"probability_{label}"] = probability[:, index]
    predictions.to_csv(out / "validation_predictions.csv", index=False, encoding="utf-8-sig")
    baseline_columns = ["source_file_id", "source_row", "rfid_id", "fishing_code", "prediction"]
    old = pd.read_csv(previous / "validation_predictions.csv", usecols=baseline_columns)
    print("동일 관측 정합성 확인 및 선박 bootstrap 2,000회", flush=True)
    bootstrap = compare.paired_bootstrap(old, predictions[baseline_columns], labels)
    baseline = read_json(previous / "metrics.json")["selected"]
    for key in ["accuracy", "macro_f1"]:
        if not np.isclose(bootstrap[key]["previous"], baseline[key], rtol=0, atol=1e-10):
            raise ValueError("Baseline prediction metrics differ")
    save_json(out / "paired_vessel_bootstrap.json", bootstrap)
    shutil.copyfile(model_path, out / "model.txt")
    if sha256(out / "model.txt") != stamp["model_sha256"]:
        raise ValueError("Checkpoint copy integrity failed")
    save_json(out / "model_metadata.json", {
        "classes": labels, "features": columns, "model": "model.txt",
        "source_model": str(model_path.relative_to(ROOT)), "sha256": stamp["model_sha256"],
        "config": config["config"]["candidates"]["control"],
        "cap_per_vessel": 256, "train_rows": 1044833, "seed": 42,
        "history_strength": 0, "ensemble": False,
        "selection_scope": "Best observed on previously inspected development vessel validation",
    })
    metrics = {"baseline": baseline, "selected": selected, "classes": labels,
               "valid_rows": len(rows), "vessel_overlap": 0, "independent_test": False}
    save_json(out / "metrics.json", metrics)
    write_report(out, metrics, bootstrap)
    print(json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
