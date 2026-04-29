import numpy as np
import joblib
from collections import deque, Counter
from pathlib import Path


class GesturePredictor:
    def __init__(
        self,
        payload_path: str | Path,
        smoother_n: int= 5,
    ):
        payload= joblib.load(payload_path)

        self._model= payload["model"]
        self._feat_cols= payload["feat_cols"]
        self._idx_to_label= payload["idx_to_label"]

        self._smoother_n= smoother_n
        self._buffer= deque(maxlen= smoother_n)

    def predict(self, feature_vector: np.ndarray) -> tuple[int | None, float | None]:
        raw_idx= int(self._model.predict(feature_vector)[0])
        proba= self._model.predict_proba(feature_vector)[0]
        confidence= float(proba.max())

        gesture_label= self._idx_to_label[raw_idx]
        self._buffer.append(gesture_label)

        if len(self._buffer) < self._smoother_n:
            return None, None

        smoothed= Counter(self._buffer).most_common(1)[0][0]
        return int(smoothed), confidence

    def reset(self) -> None:
        self._buffer.clear()

    @property
    def feat_cols(self) -> list:
        return self._feat_cols