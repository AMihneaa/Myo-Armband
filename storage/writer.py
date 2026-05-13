import numpy as np
import h5py

from pathlib import Path
from datetime import date

from config import HDF5_DIR, HDF5_FILENAME_TEMPLATE, GESTURE_LABELS


class TrialWriter:
    def __init__(
            self,
            subject_id: int,
            session_id: int,
    ) -> None:
        self._subject_id= subject_id
        self._session_id= session_id

        out_dir= Path(HDF5_DIR) / f"session_{session_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        filename= HDF5_FILENAME_TEMPLATE.format(
            subject_id=subject_id,
            session_id=session_id,
        )
        self._path= out_dir / filename
        self._file= h5py.File(self._path, "a")

    def save(
            self,
            emg: np.ndarray,
            imu: np.ndarray,
            gesture_id: int,
            trial_id: int,
    ) -> None:
        gesture_name= GESTURE_LABELS[gesture_id]

        group_path= f"subject_{self._subject_id:02d}/session_{self._session_id}/gesture_{gesture_name}/trial_{trial_id:02d}"
        group= self._file.require_group(group_path)

        if "emg" in group:
            del group["emg"]
        if "imu" in group:
            del group["imu"]

        group.create_dataset("emg", data=emg.astype(np.float32))
        group.create_dataset("imu", data=imu.astype(np.float32))

        group.attrs["subject_id"]= self._subject_id
        group.attrs["session_id"]= self._session_id
        group.attrs["gesture_id"]= gesture_id
        group.attrs["gesture_name"]= gesture_name
        group.attrs["trial_id"]= trial_id
        group.attrs["date"]= str(date.today())

    def close(self) -> None:
        self._file.close()