import numpy as np
import argparse

from pathlib import Path

from emg.hdf5_loader import load_dataset
from config import HDF5_DIR
from features.utils import make_feature_names

BP_BANDS= [(10.0, 25.0), (25.0, 40.0), (40.0, 65.0), (65.0, 95.0)]
IEVD_K= 10


def main(args: argparse.Namespace) -> None:
    feat_cols= make_feature_names(n_channels=8, bp_bands=BP_BANDS, ievd_k=IEVD_K)

    out_dir= Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for session_id in args.sessions:
        print(f"\nProcessing session {session_id}...")

        h5_dir= Path(HDF5_DIR) / f"session_{session_id}"
        X, y, meta= load_dataset(
            h5_dir=h5_dir,
            session_id=session_id,
            feat_cols=feat_cols,
        )

        np.save(out_dir / f"X_session_{session_id}.npy", X)
        np.save(out_dir / f"y_session_{session_id}.npy", y)
        meta.to_csv(out_dir / f"meta_session_{session_id}.csv", index=False)

        print(f"Session {session_id} — X: {X.shape}, y: {y.shape}")
        print(f"Classes: {dict(zip(*np.unique(y, return_counts=True)))}")
        print(f"Saved to {out_dir}")


if __name__ == "__main__":
    parser= argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data_features")
    parser.add_argument("--sessions", type=int, nargs="+", default=[1])
    args= parser.parse_args()

    main(args)