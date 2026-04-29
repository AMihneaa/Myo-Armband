import numpy as np
from numpy.linalg import svd
from numpy.lib.stride_tricks import sliding_window_view


class IEVDFeatureExtractor:
    def __init__(self, signal, L=25, k=5):
        self.signal = signal
        self.L = L
        self.k = k

    def _sigma_vec(self, x):
        N = len(x)
        L = min(self.L, max(2, N - 1))

        if L >= N:
            return np.zeros(self.k, dtype=np.float32)

        H = sliding_window_view(x, L).T
        s = svd(H, compute_uv=False)

        s = s / (s.sum() + 1e-18)
        s = np.log1p(s)

        if len(s) >= self.k:
            s = s[:self.k]
        else:
            s = np.pad(s, (0, self.k - len(s)))

        return s.astype(np.float32)

    def apply_ievd_and_extract_features(self):
        out = []
        for ch in range(self.signal.shape[1]):
            s = self._sigma_vec(self.signal[:, ch])
            out.extend(s.tolist())
        return np.asarray(out, dtype=np.float32)