import numpy as np
from scipy.signal import welch
from scipy.integrate import trapezoid


class BandPowerFeatureExtractor:
    def __init__(self, signal, fs=200, bands=None, log_power=True, with_ratios=True):
        self.signal = signal
        self.fs = fs
        self.log_power = log_power
        self.with_ratios = with_ratios

        if bands is None:
            self.bands = [(10, 25), (25, 40), (40, 65), (65, 95)]
        else:
            self.bands = bands

        nyq = fs / 2.0
        self.bands = [(lo, min(hi, nyq - 1e-6)) for lo, hi in self.bands if lo < nyq]

    def extract_features(self):
        feats = []

        for ch in range(self.signal.shape[1]):
            x = self.signal[:, ch]
            nper = len(x)

            f, Pxx = welch(
                x,
                fs=self.fs,
                nperseg=nper,
                noverlap=0,
                detrend=False,
            )

            Pxx = np.maximum(Pxx, 1e-18)
            total = trapezoid(Pxx, f)

            for lo, hi in self.bands:
                m = (f >= lo) & (f < hi)
                p = trapezoid(Pxx[m], f[m]) if np.any(m) else 0.0

                val = np.log10(p + 1e-18) if self.log_power else p
                feats.append(val)

                if self.with_ratios:
                    feats.append((p / total) if total > 0 else 0.0)

        return np.asarray(feats, dtype=np.float32)