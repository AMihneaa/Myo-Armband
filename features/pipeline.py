import numpy as np
import joblib
from pathlib import Path

from features.statistical import StatisticalFeatureExtractor
from features.band_power import BandPowerFeatureExtractor
from features.ievd import IEVDFeatureExtractor
from features.utils import make_feature_names


class LiveFeaturePipeline:
    def __init__(
        self,
        payload_path: str | Path,
        fs: float= 200.0,
        bp_bands: list= None,
        ievd_L: int= 25,
        ievd_k: int= 10,
    ):
        self.fs= fs
        self.bp_bands= bp_bands if bp_bands is not None else [(10.0, 25.0), (25.0, 40.0), (40.0, 65.0), (65.0, 95.0)]
        self.ievd_L= ievd_L
        self.ievd_k= ievd_k

        payload= joblib.load(payload_path)
        self.feat_cols= payload["feat_cols"]

        full_feat_names= make_feature_names(
            n_channels= 8,
            bp_bands= self.bp_bands,
            ievd_k= ievd_k,
        )

        name_to_idx= {name: i for i, name in enumerate(full_feat_names)}

        missing= [name for name in self.feat_cols if name not in name_to_idx]
        if missing:
            raise ValueError(f"feat_cols contains names not in full feature list: {missing}")

        self._feat_indices= np.array(
            [name_to_idx[name] for name in self.feat_cols],
            dtype= np.int32,
        )

    def extract(self, raw_window: np.ndarray, env_window: np.ndarray) -> np.ndarray:
        stat= StatisticalFeatureExtractor(raw_window, env_window, self.fs).extract_features()
        bp= BandPowerFeatureExtractor(raw_window, self.fs, bands= self.bp_bands).extract_features()
        ievd= IEVDFeatureExtractor(raw_window, L= self.ievd_L, k= self.ievd_k).apply_ievd_and_extract_features()

        full_vec= np.concatenate([stat, bp, ievd])

        return full_vec[self._feat_indices].reshape(1, -1).astype(np.float32)