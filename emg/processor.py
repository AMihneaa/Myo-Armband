import numpy as np
from fractions import Fraction
from scipy.signal import butter, filtfilt, iirnotch, resample_poly


class EMGProcessor:
    def __init__(
        self,
        file_base=None,
        fs_original=None,
        fs_target=200,
        env_cutoff=10.0,
        channels=None,
        bp_low=10.0,
        bp_high=95.0,
        notch_freq=50.0,
        notch_q=30,
        apply_notch=True,
        zscore=False,
    ):
        self.file_base = file_base
        self.fs_original = fs_original
        self.fs_target = fs_target
        self.env_cutoff = env_cutoff
        self.channels = channels

        self.bp_low = bp_low
        self.bp_high = bp_high
        self.notch_freq = notch_freq
        self.notch_q = notch_q
        self.apply_notch = apply_notch
        self.zscore = zscore

    def _butter_bandpass(self, lowcut, highcut, order=4, fs=200.0):
        nyq = 0.5 * fs
        if not (0 < lowcut < highcut < nyq):
            raise ValueError(
                f"Bandpass invalid: lowcut={lowcut}, highcut={highcut}, nyquist={nyq}"
            )
        b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
        return b, a

    def _butter_lowpass(self, cutoff=10.0, order=4, fs=200.0):
        nyq = 0.5 * fs
        if not (0 < cutoff < nyq):
            raise ValueError(f"Lowpass invalid: cutoff={cutoff}, nyquist={nyq}")
        b, a = butter(order, cutoff / nyq, btype="low")
        return b, a

    def notch_filter(self, signal, notch_freq=50.0, Q=30, fs=200.0):
        if notch_freq >= fs / 2:
            return signal
        b, a = iirnotch(notch_freq / (fs / 2), Q)
        return filtfilt(b, a, signal, axis=0)

    def _zscore_per_channel(self, X):
        return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    def process_array(self, X, fs_original=None):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array (samples, channels), got shape={X.shape}")

        fs_orig = float(fs_original if fs_original is not None else self.fs_original)
        if fs_orig <= 0:
            raise ValueError("fs_original must be > 0")

        if self.channels is not None:
            X = X[:, self.channels]

        if self.apply_notch:
            X = self.notch_filter(
                X,
                notch_freq=self.notch_freq,
                Q=self.notch_q,
                fs=fs_orig,
            )

        lowcut = self.bp_low
        highcut = min(self.bp_high, 0.49 * fs_orig, 0.48 * self.fs_target)

        if highcut <= lowcut + 1.0:
            raise ValueError(
                f"Impossible band-pass setup: lowcut={lowcut}, highcut={highcut}, "
                f"fs_orig={fs_orig}, fs_target={self.fs_target}"
            )

        b, a = self._butter_bandpass(lowcut=lowcut, highcut=highcut, order=4, fs=fs_orig)
        X = filtfilt(b, a, X, axis=0)

        fs_from = int(round(fs_orig))
        fs_to = int(round(self.fs_target))

        if fs_from != fs_to:
            frac = Fraction(fs_to, fs_from).limit_denominator()
            X = resample_poly(X, up=frac.numerator, down=frac.denominator, axis=0)
            fs = fs_to
        else:
            fs = fs_from

        env = np.abs(X)
        b, a = self._butter_lowpass(cutoff=self.env_cutoff, order=4, fs=fs)
        env = filtfilt(b, a, env, axis=0)

        if self.zscore:
            X = self._zscore_per_channel(X)
            env = self._zscore_per_channel(env)

        return X.astype(np.float32), env.astype(np.float32), fs
