import logging
import numpy as np
from pathlib import Path

from emg.live_processor import LiveEMGProcessor
from features.pipeline import LiveFeaturePipeline
from inference.predictor import GesturePredictor

logger= logging.getLogger(__name__)


class LiveInferencePipeline:
    def __init__(
        self,
        payload_path: str | Path,
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
        bp_bands: list= None,
        ievd_L: int= 25,
        ievd_k: int= 10,
        smoother_n: int= 5,
    ):
        payload_path= Path(payload_path)

        self._processor= LiveEMGProcessor(
            fs= fs,
            bp_low= bp_low,
            bp_high= bp_high,
            env_cutoff= env_cutoff,
            notch_freq= notch_freq,
            notch_q= notch_q,
            apply_notch= apply_notch,
            n_channels= n_channels,
            win= win,
            hop= hop,
        )

        self._feature_pipeline= LiveFeaturePipeline(
            payload_path= payload_path,
            fs= fs,
            bp_bands= bp_bands,
            ievd_L= ievd_L,
            ievd_k= ievd_k,
        )

        self._predictor= GesturePredictor(
            payload_path= payload_path,
            smoother_n= smoother_n,
        )

        self._window_count= 0

    def on_emg(self, chunk: np.ndarray) -> None:
        chunk= np.asarray(chunk, dtype=np.float32)

        for raw_w, env_w in self._processor.push(chunk):
            self._window_count += 1

            feat= self._feature_pipeline.extract(raw_w, env_w)
            logger.debug("window=%d feat_shape=%s", self._window_count, feat.shape)

            gesture, confidence= self._predictor.predict(feat)

            if gesture is not None:
                logger.info("gesture=%d confidence=%.4f window=%d", gesture, confidence, self._window_count)
                self._on_prediction(gesture, confidence)

    def _on_prediction(self, gesture: int, confidence: float) -> None:
        print(f"Gesture: {gesture:2d}  Confidence: {confidence:.4f}")

    def reset(self) -> None:
        self._processor.reset()
        self._predictor.reset()
        self._window_count= 0
        logger.debug("LiveInferencePipeline reset")