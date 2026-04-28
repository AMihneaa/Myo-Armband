import numpy as np
from scipy.signal import welch
from scipy.integrate import trapezoid, cumulative_trapezoid


class StatisticalFeatureExtractor:
    def __init__(
        self,
        signal_raw,
        signal_env,
        fs=200,
        zc_thresh_scale=0.01,
        ssc_thresh_scale=0.01,
        wamp_thresh_scale=0.02,
    ):
        self.raw = signal_raw
        self.env = signal_env
        self.fs = fs

        self.zc_thresh_scale = zc_thresh_scale
        self.ssc_thresh_scale = ssc_thresh_scale
        self.wamp_thresh_scale = wamp_thresh_scale

    def _zero_crossings(self, x, thr):
        x1 = x[:-1]
        x2 = x[1:]
        return int(np.sum(((x1 * x2) < 0) & (np.abs(x1 - x2) >= thr)))

    def _slope_sign_changes(self, x, thr):
        x_prev = x[:-2]
        x_curr = x[1:-1]
        x_next = x[2:]

        d1 = x_curr - x_prev
        d2 = x_curr - x_next

        return int(np.sum(((d1 * d2) > 0) & ((np.abs(d1) >= thr) | (np.abs(d2) >= thr))))

    def _willison_amplitude(self, x, thr):
        return int(np.sum(np.abs(np.diff(x)) >= thr))

    def extract_features(self):
        feats = []

        for ch in range(self.raw.shape[1]):
            xr = self.raw[:, ch]
            xe = self.env[:, ch]

            sigma = float(np.std(xr)) + 1e-8
            zc_thr = self.zc_thresh_scale * sigma
            ssc_thr = self.ssc_thresh_scale * sigma
            wamp_thr = self.wamp_thresh_scale * sigma

            env_rms = float(np.sqrt(np.mean(xe ** 2)))
            env_mav = float(np.mean(np.abs(xe)))

            raw_rms = float(np.sqrt(np.mean(xr ** 2)))
            raw_mav = float(np.mean(np.abs(xr)))
            iemg = float(np.sum(np.abs(xr)))
            ssi = float(np.sum(xr ** 2))
            var = float(np.var(xr))
            wl = float(np.sum(np.abs(np.diff(xr))))
            zc = float(self._zero_crossings(xr, zc_thr))
            ssc = float(self._slope_sign_changes(xr, ssc_thr))
            wamp = float(self._willison_amplitude(xr, wamp_thr))
            log_rms = float(np.log(raw_rms + 1e-8))

            nper = len(xr)
            f, Pxx = welch(
                xr,
                fs=self.fs,
                nperseg=nper,
                noverlap=0,
                detrend=False,
            )
            Pxx = np.maximum(Pxx, 1e-18)

            area = float(trapezoid(Pxx, f))
            mnf = float(trapezoid(f * Pxx, f) / area) if area > 0 else 0.0

            cdf = cumulative_trapezoid(Pxx, f, initial=0.0)
            half = 0.5 * cdf[-1]

            if half == 0.0 or len(cdf) == 0:
                mdf = float(f[len(f) // 2])
            else:
                idx = int(np.searchsorted(cdf, half))
                idx = np.clip(idx, 0, len(f) - 1)
                mdf = float(f[idx])

            feats.extend([
                env_rms,
                env_mav,
                raw_rms,
                raw_mav,
                iemg,
                ssi,
                var,
                wl,
                zc,
                ssc,
                wamp,
                log_rms,
                mnf,
                mdf,
            ])

        return np.asarray(feats, dtype=np.float32)