"""Compact, label-free geographic RBF embeddings fitted on training vessels."""

from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits

ANCHORS = [
    ("current", "center_lon", "center_lat"),
    ("quiet", "behavior_past_mean_quiet_lon", "behavior_past_mean_quiet_lat"),
]


class SpatialEmbedding:
    """Eight learned geographic anchors; 16 affinities and two missing flags."""

    def __init__(self, n_centers: int = 8) -> None:
        self.n_centers = n_centers

    @staticmethod
    def sphere(coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        coordinates = np.asarray(coordinates, dtype="float64")
        valid = np.isfinite(coordinates).all(axis=1)
        valid &= (np.abs(coordinates[:, 0]) <= 180) & (np.abs(coordinates[:, 1]) <= 90)
        longitude, latitude = np.radians(np.where(valid[:, None], coordinates, 0)).T
        xyz = np.column_stack([np.cos(latitude) * np.cos(longitude),
                               np.cos(latitude) * np.sin(longitude), np.sin(latitude)])
        return xyz, valid

    def fit(self, coordinates: np.ndarray) -> "SpatialEmbedding":
        xyz, valid = self.sphere(coordinates)
        xyz = xyz[valid]
        if len(np.unique(xyz, axis=0)) < self.n_centers:
            raise ValueError("Insufficient distinct training coordinates")
        with threadpool_limits(limits=1):
            clustering = MiniBatchKMeans(n_clusters=self.n_centers, random_state=42,
                                         batch_size=2048, n_init=3).fit(xyz)
        self.centers = clustering.cluster_centers_
        self.centers /= np.linalg.norm(self.centers, axis=1, keepdims=True)
        distances = np.linalg.norm(xyz[:, None] - self.centers, axis=2) * 6371.0088
        nearest = distances.argmin(axis=1)
        self.widths_km = np.asarray([
            max(5., float(np.median(distances[nearest == index, index])))
            if np.any(nearest == index) else 5. for index in range(self.n_centers)
        ])
        return self

    @property
    def feature_names(self) -> list[str]:
        return [name for anchor, _, _ in ANCHORS for name in
                [*[f"spatial_{anchor}_rbf_{index}" for index in range(self.n_centers)],
                 f"spatial_{anchor}_missing"]]

    def transform(self, coordinates: np.ndarray) -> np.ndarray:
        xyz, valid = self.sphere(coordinates)
        distances = np.linalg.norm(xyz[:, None] - self.centers, axis=2) * 6371.0088
        affinity = np.exp(-.5 * (distances / self.widths_km) ** 2)
        affinity[~valid] = 0
        return np.column_stack([affinity, ~valid]).astype("float32")

    def save(self, path: Path) -> None:
        np.savez(path, centers=self.centers, widths_km=self.widths_km)

    @classmethod
    def load(cls, path: Path) -> "SpatialEmbedding":
        with np.load(path) as state:
            model = cls(len(state["centers"]))
            model.centers, model.widths_km = state["centers"], state["widths_km"]
        return model


def append_embedding(matrix: np.ndarray, columns: list[str], compact: np.ndarray,
                     fit_rows: np.ndarray, state_path: Path) -> tuple[np.ndarray, list[str]]:
    """Fit only provided training rows, then transform other rows in batches."""
    indices = [[columns.index(lon), columns.index(lat)] for _, lon, lat in ANCHORS]
    if state_path.exists():
        encoder = SpatialEmbedding.load(state_path)
    else:
        coordinates = np.concatenate([matrix[np.ix_(fit_rows, pair)] for pair in indices])
        encoder = SpatialEmbedding().fit(coordinates)
        encoder.save(state_path)
    result = np.empty((len(matrix), compact.shape[1] + len(encoder.feature_names)), dtype="float32")
    result[:, :compact.shape[1]] = compact
    for start in range(0, len(matrix), 50000):
        stop = min(start + 50000, len(matrix))
        result[start:stop, compact.shape[1]:] = np.column_stack([
            encoder.transform(matrix[start:stop, pair]) for pair in indices
        ])
    return result, encoder.feature_names
