"""Multiscale geographic priors, with exact leave-one-vessel-out training.

Each vessel contributes at most one vote per cell. Holdout labels are never read.
Deployment state contains cell class counts, not vessel identifiers.
"""
import json
import numpy as np
import pandas as pd


def cross_fit_prior(x, vessels, y, anchors, n_splits=5, seed=42):
    """Encode each training vessel using other folds only, then fit deployment state.

    Fold assignment is independent of labels. Group cross-fitting also avoids
    the tiny self-exclusion artifacts that leave-one-out target encodings can
    expose to a tree. Never use a training vessel's own target to encode it.
    """
    unique = np.unique(vessels)
    n_splits = min(n_splits, len(unique))
    if n_splits < 2:
        raise ValueError('Cross-fitting needs at least two vessels')
    shuffled = np.random.default_rng(seed).permutation(unique)
    fold_map = pd.Series(np.arange(len(unique)) % n_splits, index=shuffled)
    fold = fold_map.reindex(vessels).to_numpy()
    result = None
    for i in range(n_splits):
        train, valid = np.flatnonzero(fold != i), np.flatnonzero(fold == i)
        assert not set(vessels[train]) & set(vessels[valid])
        encoder = SpatialPrior(anchors).fit(x.iloc[train], vessels[train], y[train])
        encoded = encoder.transform(x.iloc[valid])
        if result is None:
            result = pd.DataFrame(np.nan, index=x.index, columns=encoded.columns, dtype='float32')
        result.iloc[valid] = encoded.to_numpy()
    return result, SpatialPrior(anchors).fit(x, vessels, y)


class SpatialPrior:
    def __init__(self, anchors, scales=(.005, .02, .1), strength=5., n_classes=4):
        self.anchors = anchors
        self.scales = list(scales)
        self.strength = strength
        self.n_classes = n_classes

    @staticmethod
    def cells(xy, scale):
        valid = np.isfinite(xy).all(axis=1)
        keys = np.full(len(xy), np.iinfo('int64').min, dtype='int64')
        ij = np.floor(xy[valid] / scale).astype('int64')
        keys[valid] = (ij[:, 0] << 32) | (ij[:, 1] & 0xffffffff)
        return keys, valid

    def fit(self, x, vessels, y):
        unique = pd.DataFrame({'vessel': vessels, 'label': y}).drop_duplicates()
        if not unique.vessel.is_unique:
            raise ValueError('Conflicting vessel labels')
        self.vessel_labels = unique.set_index('vessel').label
        self.prior_counts = np.bincount(unique.label, minlength=self.n_classes).astype('float64')
        self.tables = []
        for name, lon, lat in self.anchors:
            for scale in self.scales:
                keys, valid = self.cells(x[[lon, lat]].to_numpy(), scale)
                votes = pd.DataFrame({'cell': keys[valid], 'vessel': vessels[valid], 'label': y[valid]}).drop_duplicates(['cell', 'vessel'])
                counts = votes.groupby(['cell', 'label']).size().unstack(fill_value=0).reindex(columns=range(self.n_classes), fill_value=0)
                self.tables.append({'anchor': name, 'lon': lon, 'lat': lat, 'scale': scale,
                                    'keys': counts.index.to_numpy('int64'), 'counts': counts.to_numpy('int32'),
                                    'members': pd.MultiIndex.from_frame(votes[['cell', 'vessel']])})
        return self

    def transform(self, x, vessels=None, leave_one_vessel_out=False):
        global_counts = np.broadcast_to(self.prior_counts, (len(x), self.n_classes)).copy()
        if leave_one_vessel_out:
            if not hasattr(self, 'vessel_labels') or vessels is None:
                raise ValueError('LOO encoding requires fitted vessel membership')
            own_labels = self.vessel_labels.reindex(vessels).to_numpy()
            if pd.isna(own_labels).any():
                raise ValueError('LOO rows must belong to fitted vessels')
            own_labels = own_labels.astype('int32')
            global_counts[np.arange(len(x)), own_labels] -= 1
        totals = global_counts.sum(axis=1, keepdims=True)
        prior = np.divide(global_counts, totals, out=np.full_like(global_counts, 1 / self.n_classes), where=totals > 0)
        outputs, names = [], []
        for table in self.tables:
            keys, valid = self.cells(x[[table['lon'], table['lat']]].to_numpy(), table['scale'])
            indices = pd.Index(table['keys']).get_indexer(keys)
            known = indices >= 0
            counts = np.zeros((len(x), self.n_classes), dtype='float64')
            counts[known] = table['counts'][indices[known]]
            if leave_one_vessel_out:
                membership = pd.MultiIndex.from_arrays([keys, vessels]).isin(table['members'])
                counts[np.flatnonzero(membership), own_labels[membership]] -= 1
            support = counts.sum(axis=1)
            probabilities = (counts + self.strength * prior) / (support[:, None] + self.strength)
            prefix = f'geo_{table["anchor"]}_{table["scale"]}'
            outputs.extend([probabilities, np.log1p(support)[:, None], valid.astype('float32')[:, None]])
            names.extend([f'{prefix}_p{i}' for i in range(self.n_classes)] + [f'{prefix}_support', f'{prefix}_available'])
        return pd.DataFrame(np.column_stack(outputs).astype('float32'), index=x.index, columns=names)

    def save(self, path):
        meta = {'anchors': self.anchors, 'scales': self.scales, 'strength': self.strength,
                'n_classes': self.n_classes,
                'tables': [{k: t[k] for k in ['anchor', 'lon', 'lat', 'scale']} for t in self.tables]}
        arrays = {'metadata': np.array(json.dumps(meta)), 'prior_counts': self.prior_counts}
        for i, table in enumerate(self.tables):
            arrays[f'keys_{i}'], arrays[f'counts_{i}'] = table['keys'], table['counts']
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data['metadata']))
            obj = cls(meta['anchors'], meta['scales'], meta['strength'], meta['n_classes'])
            obj.prior_counts = data['prior_counts'].copy()
            obj.tables = [{**t, 'keys': data[f'keys_{i}'].copy(), 'counts': data[f'counts_{i}'].copy()}
                          for i, t in enumerate(meta['tables'])]
        return obj
