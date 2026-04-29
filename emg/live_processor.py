import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi, iirnotch, tf2sos


class LiveEMGProcessor:
    def __init__(
        self,
        fs: float= 200.0,
        bp_low: float= 10.0,
        bp_high: float= 95.0,
        env_cutoff: float= 10.0,
        notch_freq: float= 50.0,
        notch_q: float= 30.0,
        apply_notch: bool= True,
        n_channels: int= 8,
        win: int= 50,
        hop: int= 25,
    ):
        self.fs= fs
        self.bp_low= bp_low
        self.bp_high= bp_high
        self.env_cutoff= env_cutoff
        self.notch_freq= notch_freq
        self.notch_q= notch_q
        self.apply_notch= apply_notch
        self.n_channels= n_channels
        self.win= win
        self.hop= hop

        nyq= fs / 2.0

        self._sos_bp= butter(
            4,
            [bp_low / nyq, min(bp_high, nyq - 1e-6) / nyq],
            btype="band",
            output="sos",
        )

        self._sos_lp= butter(
            4,
            env_cutoff / nyq,
            btype="low",
            output="sos",
        )

        if apply_notch and notch_freq < nyq:
            b, a= iirnotch(notch_freq / nyq, notch_q)
            self._sos_notch= tf2sos(b, a)
        else:
            self._sos_notch= None

        self._zi_bp= self._make_zi(self._sos_bp)
        self._zi_lp= self._make_zi(self._sos_lp)
        self._zi_notch= self._make_zi(self._sos_notch) if self._sos_notch is not None else None

        buffer_size= win * 4
        self._raw_buffer= np.zeros((buffer_size, n_channels), dtype=np.float32)
        self._env_buffer= np.zeros((buffer_size, n_channels), dtype=np.float32)
        self._buffer_size= buffer_size
        self._pos= 0
        self._total= 0
        self._new_samples= -(self.win - self.hop)

    def push(self, chunk: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        chunk= np.asarray(chunk, dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[1] != self.n_channels:
            raise ValueError(
                f"Expected chunk shape (n_samples, {self.n_channels}), got {chunk.shape}"
            )

        n= chunk.shape[0]

        if self._sos_notch is not None:
            chunk, self._zi_notch= sosfilt(
                self._sos_notch, chunk, axis=0, zi=self._zi_notch
            )

        filtered, self._zi_bp= sosfilt(
            self._sos_bp, chunk, axis=0, zi=self._zi_bp
        )

        env, self._zi_lp= sosfilt(
            self._sos_lp, np.abs(filtered), axis=0, zi=self._zi_lp
        )

        for i in range(n):
            self._raw_buffer[self._pos]= filtered[i]
            self._env_buffer[self._pos]= env[i]
            self._pos= (self._pos + 1) % self._buffer_size

        self._total += n
        self._new_samples += n

        results= []
        while self._new_samples >= self.hop and self._total >= self.win:
            self._new_samples -= self.hop
            results.append(self._extract_window())

        return results

    def reset(self) -> None:
        self._zi_bp= self._make_zi(self._sos_bp)
        self._zi_lp= self._make_zi(self._sos_lp)
        self._zi_notch= (
            self._make_zi(self._sos_notch)
            if self._sos_notch is not None
            else None
        )

        self._raw_buffer[:]= 0.0
        self._env_buffer[:]= 0.0
        self._pos= 0
        self._total= 0
        self._new_samples= -(self.win - self.hop)


    def _make_zi(self, sos: np.ndarray) -> np.ndarray:
        zi_base= sosfilt_zi(sos)
        zi= zi_base[:, :, np.newaxis] * np.ones((1, 1, self.n_channels))
        return zi

    def _extract_window(self) -> tuple[np.ndarray, np.ndarray]:
        indices= [(self._pos - self.win + i) % self._buffer_size for i in range(self.win)]
        raw_window= self._raw_buffer[indices].copy()
        env_window= self._env_buffer[indices].copy()
        return raw_window, env_window
