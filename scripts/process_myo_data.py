import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from emg.processor import EMGProcessor
from emg.myo_adapter import load_myo_parquet
from features.statistical import StatisticalFeatureExtractor
from features.band_power import BandPowerFeatureExtractor
from features.ievd import IEVDFeatureExtractor
from features.utils import iter_windows, make_feature_names

FS= 200.0
WIN= 50
HOP= 25
IEVD_L= 25
IEVD_K= 10
BP_BANDS= [(10.0, 25.0), (25.0, 40.0), (40.0, 65.0), (65.0, 95.0)]
BP_LOW= 10.0
BP_HIGH= 95.0
ENV_CUTOFF= 10.0

META_COLS= ["participant", "session", "gesture", "trial", "win_idx", "t_start", "t_end", "fs"]


def process_file(parquet_path, feat_cols) -> pd.DataFrame:
    processor= EMGProcessor(fs_target=FS, bp_low=BP_LOW, bp_high=BP_HIGH, env_cutoff=ENV_CUTOFF)
    raw, env, fs= load_myo_parquet(parquet_path=parquet_path, processor=processor, fs=FS)

    meta= pd.read_parquet(parquet_path)[["participant", "session", "gesture", "trial"]]
    participant, session, gesture, trial= meta.iloc[0]

    N= raw.shape[0]
    if N < WIN:
        raise ValueError(f"Signal too short: {N} samples, need at least {WIN}")

    feat_names= make_feature_names(n_channels=8, bp_bands=BP_BANDS, ievd_k=IEVD_K)

    rows_meta= []
    rows_feat= []

    for win_idx, (a, b) in enumerate(iter_windows(N=N, win=WIN, hop=HOP)):
        w_raw= raw[a:b, :]
        w_env= env[a:b, :]  

        stat= StatisticalFeatureExtractor(w_raw, w_env, fs).extract_features()
        bp= BandPowerFeatureExtractor(w_raw, fs, bands=BP_BANDS).extract_features()
        ievd= IEVDFeatureExtractor(w_raw, L=IEVD_L, k=IEVD_K).apply_ievd_and_extract_features()

        feat_vec= np.concatenate([stat, bp, ievd]).astype(np.float32)

        rows_feat.append(feat_vec)
        rows_meta.append({
            "participant": int(participant),
            "session": int(session),
            "gesture": int(gesture),
            "trial": int(trial),
            "win_idx": win_idx,
            "t_start": float(a / fs),
            "t_end": float(b / fs),
            "fs": int(fs),
        })

    feat_mat= np.vstack(rows_feat)
    df_meta= pd.DataFrame(rows_meta)
    df_feat= pd.DataFrame(feat_mat, columns=feat_names)
    df= pd.concat([df_meta, df_feat], axis=1)

    return df[META_COLS + feat_cols]


if __name__ == "__main__":
    parquet_path= Path("data_raw/session1/participant1/session1_participant1_gesture1_trial1.parquet")
    out_path= Path("data_processed/p1_s1_g1_t1.parquet")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    model= joblib.load("models/xgb_gpu_search_200hz_g1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17_trainSession1_valSession2_testSession3_20260326_160454_trainval_final.pkl")
    feat_cols= model["feat_cols"]

    df= process_file(parquet_path, feat_cols)
    df.to_parquet(out_path, index=False)

    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"NaN: {df.isnull().sum().sum()}")