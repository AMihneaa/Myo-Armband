import h5py
import numpy as np
import pandas as pd

from pathlib import Path

from emg.processor import EMGProcessor
from features.statistical import StatisticalFeatureExtractor
from features.band_power import BandPowerFeatureExtractor
from features.ievd import IEVDFeatureExtractor
from features.utils import iter_windows, make_feature_names

from config import (
    SAMPLE_RATE,
    BP_LOW,
    BP_HIGH,
    ENV_CUTOFF,
    NOTCH_FREQ,
    NOTCH_Q,
    APPLY_NOTCH,
    GESTURE_LABELS,
)

WIN= 50
HOP= 25
IEVD_L= 25
IEVD_K= 10
BP_BANDS= [(10.0, 25.0), (25.0, 40.0), (40.0, 65.0), (65.0, 95.0)]


def _make_processor() -> EMGProcessor:
    return EMGProcessor(
        fs_target=SAMPLE_RATE,
        bp_low=BP_LOW,
        bp_high=BP_HIGH,
        env_cutoff=ENV_CUTOFF,
        notch_freq=NOTCH_FREQ,
        notch_q=NOTCH_Q,
        apply_notch=APPLY_NOTCH,
    )


def load_trial(
        h5_path: str | Path,
        subject: str,
        session: str,
        gesture: str,
        trial: str,
        processor: EMGProcessor,
) -> tuple[np.ndarray, np.ndarray, float]:
    with h5py.File(h5_path, "r") as f:
        emg= f[f"{subject}/{session}/{gesture}/{trial}/emg"][:]

    raw, env, fs= processor.process_array(emg, fs_original=float(SAMPLE_RATE))
    return raw, env, fs


def _process_trial(
        raw: np.ndarray,
        env: np.ndarray,
        fs: float,
        subject_id: int,
        session_id: int,
        gesture_id: int,
        trial_id: int,
        feat_names: list[str],
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    N= raw.shape[0]
    if N < WIN:
        raise ValueError(f"Signal too short: {N} samples, need at least {WIN}")

    rows_feat= []
    rows_meta= []

    for win_idx, (a, b) in enumerate(iter_windows(N=N, win=WIN, hop=HOP)):
        w_raw= raw[a:b, :]
        w_env= env[a:b, :]

        stat= StatisticalFeatureExtractor(w_raw, w_env, fs).extract_features()
        bp= BandPowerFeatureExtractor(w_raw, fs, bands=BP_BANDS).extract_features()
        ievd= IEVDFeatureExtractor(w_raw, L=IEVD_L, k=IEVD_K).apply_ievd_and_extract_features()

        feat_vec= np.concatenate([stat, bp, ievd]).astype(np.float32)
        rows_feat.append(feat_vec)
        rows_meta.append({
            "subject_id": subject_id,
            "session_id": session_id,
            "gesture_id": gesture_id,
            "gesture_name": GESTURE_LABELS[gesture_id],
            "trial_id": trial_id,
            "win_idx": win_idx,
            "t_start": float(a / fs),
            "t_end": float(b / fs),
        })

    return np.vstack(rows_feat), np.full(len(rows_feat), gesture_id, dtype=np.int32), rows_meta


def load_dataset(
        h5_dir: str | Path,
        session_id: int,
        feat_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    h5_dir= Path(h5_dir)
    processor= _make_processor()
    feat_names= make_feature_names(n_channels=8, bp_bands=BP_BANDS, ievd_k=IEVD_K)

    name_to_idx= {name: i for i, name in enumerate(feat_names)}
    feat_indices= np.array([name_to_idx[n] for n in feat_cols], dtype=np.int32)

    all_X= []
    all_y= []
    all_meta= []

    label_to_id= {v: k for k, v in GESTURE_LABELS.items()}

    h5_files= sorted(h5_dir.glob(f"*_session_{session_id}.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No HDF5 files found in {h5_dir} for session {session_id}")

    for h5_path in h5_files:
        subject_id= int(h5_path.stem.split("_")[1])
        subject_key= f"subject_{subject_id:02d}"
        session_key= f"session_{session_id}"

        with h5py.File(h5_path, "r") as f:
            if subject_key not in f or session_key not in f[subject_key]:
                print(f"Skipping {h5_path.name} — structure not found")
                continue

            gesture_keys= list(f[subject_key][session_key].keys())
            trial_keys_per_gesture= {
                g: list(f[subject_key][session_key][g].keys())
                for g in gesture_keys
            }

        for gesture_key in gesture_keys:
            gesture_name= gesture_key.replace("gesture_", "")
            if gesture_name not in label_to_id:
                print(f"Skipping unknown gesture: {gesture_key}")
                continue

            gesture_id= label_to_id[gesture_name]

            for trial_key in trial_keys_per_gesture[gesture_key]:
                trial_id= int(trial_key.split("_")[1])

                raw, env, fs= load_trial(
                    h5_path=h5_path,
                    subject=subject_key,
                    session=session_key,
                    gesture=gesture_key,
                    trial=trial_key,
                    processor=processor,
                )

                X_trial, y_trial, meta_trial= _process_trial(
                    raw=raw,
                    env=env,
                    fs=fs,
                    subject_id=subject_id,
                    session_id=session_id,
                    gesture_id=gesture_id,
                    trial_id=trial_id,
                    feat_names=feat_names,
                )

                all_X.append(X_trial[:, feat_indices])
                all_y.append(y_trial)
                all_meta.extend(meta_trial)

        print(f"Processed: {h5_path.name}")

    X= np.vstack(all_X).astype(np.float32)
    y= np.concatenate(all_y).astype(np.int32)
    meta= pd.DataFrame(all_meta)

    return X, y, meta