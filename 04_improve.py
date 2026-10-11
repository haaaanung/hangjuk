"""Reproducible vessel-disjoint model selection; preserves the week-2 baseline."""
import argparse
import gc
import json
import platform
import subprocess
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from src.common import ROOT, OUTPUT, data_files
from src.context_features import VERSION, build_context, load_locations, sha256
from src.features import FEATURE_COLUMNS


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def prepare():
    source = OUTPUT / 'feature_df.csv'
    df = pd.read_csv(source, dtype={c: 'float32' for c in FEATURE_COLUMNS})
    if df.groupby('rfid_id').fishing_code.nunique().max() != 1:
        raise ValueError('Vessel label changes need investigation')
    if df.duplicated(['rfid_id', 'part_dt', 'rfid_time']).any():
        raise ValueError('Duplicate analysis rows')
    cache = OUTPUT / 'cache' / 'context_v1'
    cache.mkdir(parents=True, exist_ok=True)
    fingerprint = sha256(source)
    locations = load_locations(df, data_files(), cache)
    signature = {'feature_version': VERSION, 'input_sha256': fingerprint,
                 'location_manifest_sha256': sha256(cache / 'locations.json'),
                 'implementation_sha256': sha256(ROOT / 'src' / 'context_features.py')}
    manifest = cache / 'features.json'
    array_path = cache / 'features.npy'
    if manifest.exists() and array_path.exists():
        stored = json.loads(manifest.read_text(encoding='utf-8'))
        if stored['signature'] == signature:
            matrix = np.load(array_path, mmap_mode='r')
            if matrix.shape == (len(df), len(stored['columns'])):
                print(f'Loaded cached context: {matrix.shape}', flush=True)
                return df, matrix, stored['columns'], signature
    print('Building strictly prior-date history...', flush=True)
    features = build_context(df, locations)
    np.save(array_path, features.to_numpy())
    columns = features.columns.tolist()
    write_json(manifest, {'signature': signature, 'columns': columns})
    del features, locations
    gc.collect()
    return df, np.load(array_path, mmap_mode='r'), columns, signature


def candidates():
    return {
        'baseline_balanced': {'features': 'baseline', 'weight_power': 1., 'rounds': 150},
        'baseline_unweighted': {'features': 'baseline', 'weight_power': 0., 'rounds': 150},
        'location_sqrt': {'features': 'location', 'weight_power': .5, 'rounds': 350},
        'history_sqrt': {'features': 'history', 'weight_power': .5, 'rounds': 350},
        'history_unweighted': {'features': 'history', 'weight_power': 0., 'rounds': 350},
    }


def feature_indices(config, columns):
    if config['features'] == 'baseline':
        return [columns.index(c) for c in FEATURE_COLUMNS]
    if config['features'] == 'location':
        return [i for i, c in enumerate(columns) if not c.startswith('past_')
                and c != 'days_since_previous_date']
    return list(range(len(columns)))


def sample_training(indices, vessels, cap, seed=42):
    if cap == 0:
        return np.asarray(indices)
    rng = np.random.default_rng(seed)
    table = pd.DataFrame({'row': indices, 'vessel': vessels[indices],
                          'order': rng.random(len(indices))})
    return np.sort(table.sort_values('order').groupby('vessel').head(cap).row.to_numpy())


def fit(config, matrix, columns, indices, y, threads):
    selected = feature_indices(config, columns)
    counts = np.bincount(y[indices])
    if (counts == 0).any():
        raise ValueError('Training split missing a class')
    weights = (len(indices) / (len(counts) * counts)) ** config['weight_power']
    classic = config['features'] == 'baseline'
    params = dict(objective='multiclass', n_estimators=config['rounds'], learning_rate=.05,
                  num_leaves=31, min_child_samples=50 if classic else 200,
                  reg_lambda=0 if classic else 5, random_state=42, n_jobs=threads,
                  verbosity=-1, importance_type='gain', deterministic=True,
                  force_col_wise=True)
    model = lgb.LGBMClassifier(**params)
    started = time.perf_counter()
    train_x = pd.DataFrame(matrix[np.ix_(indices, selected)], columns=[columns[i] for i in selected])
    model.fit(train_x, y[indices], sample_weight=weights[y[indices]])
    return model, time.perf_counter() - started


def predict(model, config, matrix, columns, indices):
    selected = feature_indices(config, columns)
    # Bound peak memory; all evaluation rows are retained.
    result = []
    for start in range(0, len(indices), 100000):
        rows = indices[start:start + 100000]
        x = pd.DataFrame(matrix[np.ix_(rows, selected)], columns=[columns[i] for i in selected])
        result.append(model.predict_proba(x))
    return np.concatenate(result)


def scores(truth, probabilities, vessels):
    predicted = probabilities.argmax(axis=1)
    inverse_counts = 1 / pd.Series(vessels).groupby(vessels).transform('size').to_numpy()
    labels = np.arange(probabilities.shape[1])
    return {'accuracy': accuracy_score(truth, predicted),
            'macro_f1': f1_score(truth, predicted, labels=labels, average='macro', zero_division=0),
            'vessel_equal_accuracy': accuracy_score(truth, predicted, sample_weight=inverse_counts),
            'vessel_equal_macro_f1': f1_score(truth, predicted, labels=labels, average='macro',
                                              sample_weight=inverse_counts, zero_division=0)}


def evaluate(name, truth, probabilities, vessels, labels, out):
    result = scores(truth, probabilities, vessels)
    predicted = probabilities.argmax(axis=1)
    codes = np.arange(len(labels))
    pd.DataFrame(classification_report(truth, predicted, labels=codes, target_names=labels,
                                      output_dict=True, zero_division=0)).T.to_csv(
        out / f'{name}_classification_report.csv', encoding='utf-8-sig')
    pd.DataFrame(confusion_matrix(truth, predicted, labels=codes), index=labels, columns=labels).to_csv(
        out / f'{name}_confusion_matrix.csv', encoding='utf-8-sig')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='context_v1')
    parser.add_argument('--cv-cap-per-vessel', type=int, default=128,
                        help='Training-only CV cap; 0 uses all rows. Final fit uses all training rows.')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in {'.', '..'}:
        parser.error('run-id must be a single directory name')
    if args.cv_cap_per_vessel < 0 or args.threads < 1:
        parser.error('cap must be nonnegative and threads must be positive')
    df, matrix, columns, signature = prepare()
    if args.prepare_only:
        print(f'Prepared {len(df):,} rows, {len(columns)} features', flush=True)
        return
    out = OUTPUT / 'experiments' / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    labels = sorted(df.fishing_code.unique())
    y = pd.Categorical(df.fishing_code, categories=labels).codes.astype('int32')
    vessels = df.rfid_id.to_numpy()
    train, valid = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(
        np.zeros(len(df)), y, groups=vessels))
    if set(vessels[train]) & set(vessels[valid]):
        raise ValueError('Holdout vessel overlap')
    # Explicitly ensure the old baseline and improved model have identical holdout rows.
    old_split = pd.read_csv(OUTPUT / 'vessel_split.csv')
    if set(old_split.loc[old_split.split == 'validation', 'rfid_id']) != set(vessels[valid]):
        raise ValueError('Existing baseline split does not match')
    pd.DataFrame({'rfid_id': np.r_[np.unique(vessels[train]), np.unique(vessels[valid])],
                  'split': ['train'] * len(np.unique(vessels[train])) +
                           ['validation'] * len(np.unique(vessels[valid]))}).to_csv(out / 'vessel_split.csv', index=False)
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    run_config = {'signature': signature, 'code_commit': commit.stdout.strip(),
                  'code_sha256': sha256(Path(__file__)), 'args': vars(args), 'classes': labels,
                  'features': columns, 'candidates': candidates(),
                  'versions': {'python': platform.python_version(), 'lightgbm': lgb.__version__,
                               'pandas': pd.__version__, 'numpy': np.__version__, 'sklearn': sklearn.__version__},
                  'train_rows': len(train), 'valid_rows': len(valid),
                  'train_vessels': len(np.unique(vessels[train])), 'valid_vessels': len(np.unique(vessels[valid])),
                  'vessel_overlap': 0, 'selection_metric': 'inner_cv_mean_macro_f1',
                  'evaluation_scope': 'Previously inspected same-period vessel holdout, not an independent final test',
                  'history_policy': 'Only observations on strictly earlier calendar dates; no labels'}
    write_json(out / 'config.json', run_config)
    cv_results = []
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    assignments = []
    for fold, (fit_local, eval_local) in enumerate(splitter.split(np.zeros(len(train)), y[train], vessels[train]), 1):
        fit_rows = sample_training(train[fit_local], vessels, args.cv_cap_per_vessel)
        eval_rows = train[eval_local]
        if set(vessels[fit_rows]) & set(vessels[eval_rows]):
            raise ValueError('CV vessel overlap')
        if len(np.unique(y[eval_rows])) != len(labels):
            raise ValueError('CV evaluation missing a class')
        assignments.append(pd.DataFrame({'rfid_id': np.unique(vessels[eval_rows]), 'fold': fold}))
        for name, config in candidates().items():
            print(f'CV {fold}/3 {name}: train={len(fit_rows):,}, validation={len(eval_rows):,}', flush=True)
            model, seconds = fit(config, matrix, columns, fit_rows, y, args.threads)
            proba = predict(model, config, matrix, columns, eval_rows)
            result = evaluate(f'cv{fold}_{name}', y[eval_rows], proba, vessels[eval_rows], labels, out)
            result.update(candidate=name, fold=fold, training_seconds=seconds,
                          train_rows=len(fit_rows), valid_rows=len(eval_rows), vessel_overlap=0)
            cv_results.append(result)
            pd.DataFrame(cv_results).to_csv(out / 'cv_results.csv', index=False)
            print(json.dumps(result), flush=True)
            del model, proba
            gc.collect()
    pd.concat(assignments).to_csv(out / 'cv_assignments.csv', index=False)
    cv_table = pd.DataFrame(cv_results)
    summary = cv_table.groupby('candidate').agg(
        macro_f1_mean=('macro_f1', 'mean'), macro_f1_std=('macro_f1', 'std'),
        accuracy_mean=('accuracy', 'mean'), training_seconds_mean=('training_seconds', 'mean'))
    summary.sort_values('macro_f1_mean', ascending=False).to_csv(out / 'cv_summary.csv')
    winner = summary.macro_f1_mean.idxmax()
    selected = candidates()[winner]
    print(f'Selected by inner CV: {winner}. Refitting on all {len(train):,} training rows.', flush=True)
    model, seconds = fit(selected, matrix, columns, train, y, args.threads)
    proba = predict(model, selected, matrix, columns, valid)
    improved = evaluate('selected', y[valid], proba, vessels[valid], labels, out)
    improved.update(training_seconds=seconds, candidate=winner)
    (out / 'model.txt').write_text(model.booster_.model_to_string(), encoding='utf-8')
    used = [columns[i] for i in feature_indices(selected, columns)]
    pd.Series(model.feature_importances_, index=used, name='gain').sort_values(ascending=False).to_csv(out / 'feature_importance.csv')
    # Load the exact existing model, including its recorded class ordering.
    baseline_meta = json.loads((OUTPUT / 'metrics' / 'baseline_metrics.json').read_text(encoding='utf-8'))
    baseline = lgb.Booster(model_str=(OUTPUT / 'lightgbm_baseline.txt').read_text(encoding='utf-8'))
    original_proba = baseline.predict(df.iloc[valid][FEATURE_COLUMNS], num_threads=args.threads)
    original_proba = original_proba[:, [baseline_meta['classes'].index(label) for label in labels]]
    original = evaluate('baseline', y[valid], original_proba, vessels[valid], labels, out)
    for metric in ['accuracy', 'macro_f1']:
        if not np.isclose(original[metric], baseline_meta[metric], atol=1e-10, rtol=0):
            raise ValueError(f'Baseline reproduction failed: {metric}')
    write_json(out / 'metrics.json', {'baseline': original, 'selected': improved,
                                     'delta_accuracy': improved['accuracy'] - original['accuracy'],
                                     'delta_macro_f1': improved['macro_f1'] - original['macro_f1'],
                                     'cv_selected': winner, 'features': used, 'classes': labels})
    predictions = df.iloc[valid][['source_file_id', 'source_row', 'rfid_id', 'part_dt', 'rfid_time', 'fishing_code']].copy()
    predictions['prediction'] = np.asarray(labels)[proba.argmax(axis=1)]
    for i, label in enumerate(labels):
        predictions[f'probability_{label}'] = proba[:, i]
    predictions.to_csv(out / 'validation_predictions.csv', index=False, encoding='utf-8')
    slices = []
    prior_idx = columns.index('past_observations')
    has_past = matrix[valid, prior_idx] > 0
    for name, mask in [('no_prior_date', ~has_past), ('has_prior_date', has_past)]:
        if mask.any():
            slices.append({'slice': name, 'rows': int(mask.sum()),
                           **scores(y[valid][mask], proba[mask], vessels[valid][mask])})
    pd.DataFrame(slices).to_csv(out / 'history_slices.csv', index=False)
    write_json(out / 'model_metadata.json', {'candidate': winner, 'features': used, 'classes': labels,
                                           'params': model.get_params(), 'signature': signature,
                                           'history_policy': run_config['history_policy']})
    print(json.dumps({'baseline': original, 'selected': improved}, ensure_ascii=False, indent=2), flush=True)
    print(f'Completed: {out}', flush=True)


if __name__ == '__main__':
    main()
