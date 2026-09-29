"""STEP 5~6: 같은 선박이 양쪽에 들어가지 않는 고정 분할로 기준 성능을 확인한다."""
import json
import time
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from src.common import OUTPUT, METRICS, FIGURES, data_files, draw_track, plt
from src.features import FEATURE_COLUMNS


def show_mistakes(predictions):
    wrong = predictions.loc[predictions.fishing_code != predictions.prediction]
    examples = wrong.sort_values('confidence', ascending=False).drop_duplicates(['fishing_code', 'prediction']).head(4).copy()
    examples['geom'] = ''
    for file_id, items in examples.groupby('source_file_id'):
        path = data_files()[int(file_id) - 1]
        targets = set(items.source_row)
        for chunk in pd.read_csv(path, usecols=['geom'], chunksize=10000):
            for idx in targets.intersection(chunk.index):
                examples.loc[(examples.source_file_id == file_id) & (examples.source_row == idx), 'geom'] = chunk.loc[idx, 'geom']
            if chunk.index[-1] >= max(targets):
                break
    examples.to_csv(OUTPUT / 'misclassified_examples.csv', index=False, encoding='utf-8-sig')
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax in axes.flat:
        ax.axis('off')
    for ax, (_, row) in zip(axes.flat, examples.iterrows()):
        ax.axis('on'); draw_track(ax, row.geom)
        ax.set_title(f'실제 {row.fishing_code} → 예측 {row.prediction}\n'
                     f'확률 {row.confidence:.2f} (미보정), 거리 {row.total_distance_m:.1f}m\n'
                     f'점 {row.num_points:.0f}, 직선성 {row.straightness:.3f}')
    fig.tight_layout(); fig.savefig(FIGURES / 'misclassified_tracks.png', dpi=160); plt.close(fig)


def main():
    df = pd.read_csv(OUTPUT / 'feature_df.csv', dtype={c: 'float32' for c in FEATURE_COLUMNS})
    assert not df.duplicated(['rfid_id', 'part_dt', 'rfid_time']).any(), '분석 단위 중복을 먼저 검토하세요.'
    assert df.groupby('rfid_id').fishing_code.nunique().max() == 1, '선박 label 변경을 먼저 검토하세요.'
    # random row split은 같은 선박의 반복 관측이 양쪽에 들어가 신규 선박 성능을 과대평가한다.
    split = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42)
    train, valid = next(split.split(df[FEATURE_COLUMNS], df.fishing_code, groups=df.rfid_id))
    train_vessel_ids = df.iloc[train].rfid_id.unique()
    valid_vessel_ids = df.iloc[valid].rfid_id.unique()
    overlap = set(train_vessel_ids) & set(valid_vessel_ids)
    assert not overlap
    labels = sorted(df.fishing_code.unique())
    assert set(df.iloc[train].fishing_code) == set(labels)
    assert set(df.iloc[valid].fishing_code) == set(labels)
    split_info = {'seed': 42, 'train_rows': len(train), 'valid_rows': len(valid),
                  'train_vessels': len(train_vessel_ids), 'valid_vessels': len(valid_vessel_ids),
                  'vessel_overlap': len(overlap), 'features': FEATURE_COLUMNS}
    print('split', json.dumps(split_info, ensure_ascii=False), flush=True)
    (METRICS / 'split.json').write_text(json.dumps(split_info, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame({'rfid_id': np.r_[train_vessel_ids, valid_vessel_ids],
                  'split': ['train']*len(train_vessel_ids) + ['validation']*len(valid_vessel_ids)}).to_csv(OUTPUT / 'vessel_split.csv', index=False)
    parts = []
    for name, indices in [('train', train), ('validation', valid)]:
        part = df.iloc[indices].groupby('fishing_code').agg(rows=('rfid_id', 'size'), vessels=('rfid_id', 'nunique'))
        part['split'] = name; parts.append(part)
    pd.concat(parts).to_csv(METRICS / 'split_class_counts.csv', encoding='utf-8-sig')
    # 식별자·날짜·시간·절대 위치·사고 정보는 입력하지 않는다. NaN은 LightGBM이 직접 처리한다.
    model = LGBMClassifier(objective='multiclass', n_estimators=150, learning_rate=.05,
                          num_leaves=31, min_child_samples=50, class_weight='balanced',
                          random_state=42, n_jobs=4, verbosity=-1, importance_type='gain',
                          deterministic=True, force_col_wise=True)
    start = time.perf_counter()
    model.fit(df.iloc[train][FEATURE_COLUMNS], df.iloc[train].fishing_code)
    training_seconds = time.perf_counter()-start
    print(f'학습 완료: {training_seconds:.1f}s', flush=True)
    probabilities = model.predict_proba(df.iloc[valid][FEATURE_COLUMNS])
    predicted = model.classes_[probabilities.argmax(axis=1)]
    truth = df.iloc[valid].fishing_code
    majority = df.iloc[train].fishing_code.mode().iloc[0]
    dummy = np.repeat(majority, len(valid))
    metrics = {
        'accuracy': accuracy_score(truth, predicted),
        'macro_f1': f1_score(truth, predicted, labels=labels, average='macro', zero_division=0),
        'majority_accuracy': accuracy_score(truth, dummy),
        'majority_macro_f1': f1_score(truth, dummy, labels=labels, average='macro', zero_division=0),
        'training_seconds': training_seconds,
        'classes': model.classes_.tolist(),
        'params': model.get_params(),
    }
    (METRICS / 'baseline_metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    report = pd.DataFrame(classification_report(truth, predicted, labels=labels, output_dict=True, zero_division=0)).T
    report.to_csv(METRICS / 'classification_report.csv', encoding='utf-8-sig')
    cm = confusion_matrix(truth, predicted, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(METRICS / 'confusion_matrix.csv', encoding='utf-8-sig')
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, values, fmt, title in zip(axes, [cm, cm/cm.sum(axis=1, keepdims=True)], ['d', '.2f'], ['관측 행 수', '실제 클래스별 비율 (대각선=Recall)']):
        ConfusionMatrixDisplay(values, display_labels=labels).plot(ax=ax, cmap='Blues', values_format=fmt, colorbar=False)
        ax.set_title(title); ax.tick_params(axis='x', rotation=15)
    fig.tight_layout(); fig.savefig(FIGURES / 'confusion_matrix.png', dpi=160); plt.close(fig)
    importance = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS, name='gain').sort_values(ascending=False)
    importance.to_csv(METRICS / 'feature_importance.csv')
    importance.head(12).sort_values().plot.barh(figsize=(10, 6), title='LightGBM 분할 gain: 인과관계가 아님')
    plt.tight_layout(); plt.savefig(FIGURES / 'feature_importance.png', dpi=160); plt.close('all')
    # Windows 한글 경로는 C 라이브러리의 파일 열기 대신 Python의 UTF-8 쓰기를 사용한다.
    (OUTPUT / 'lightgbm_baseline.txt').write_text(model.booster_.model_to_string(), encoding='utf-8')
    result = df.iloc[valid].copy()
    result['prediction'] = predicted; result['confidence'] = probabilities.max(axis=1)
    for i, label in enumerate(model.classes_):
        result[f'probability_{label}'] = probabilities[:, i]
    result.to_csv(OUTPUT / 'validation_predictions.csv', index=False, encoding='utf-8', float_format='%.7g')
    # 관측량별 오류는 다음 주의 시간대 집계 실험 근거로만 사용한다.
    result['point_bin'] = pd.cut(result.num_points, [0, 5, 20, 100, np.inf], labels=['2-5', '6-20', '21-100', '101+'])
    slices = []
    for name, group in result.groupby('point_bin', observed=True):
        slices.append({'point_bin': str(name), 'rows': len(group), 'accuracy': accuracy_score(group.fishing_code, group.prediction),
                       'macro_f1': f1_score(group.fishing_code, group.prediction, labels=labels, average='macro', zero_division=0)})
    pd.DataFrame(slices).to_csv(METRICS / 'performance_by_point_count.csv', index=False)
    errors = [(labels[i], labels[j], int(cm[i,j]), float(cm[i,j]/cm[i].sum()))
              for i in range(len(labels)) for j in range(len(labels)) if i != j]
    pd.DataFrame(sorted(errors, key=lambda x:x[2], reverse=True), columns=['actual', 'predicted', 'rows', 'class_fraction']).to_csv(METRICS / 'confusion_pairs.csv', index=False, encoding='utf-8-sig')
    print(report.to_string(), flush=True)
    print('Accuracy', metrics['accuracy'], 'Macro-F1', metrics['macro_f1'], flush=True)
    show_mistakes(result)


if __name__ == '__main__':
    main()
