import numpy as np
import pandas as pd

from pathlib import Path

class TrialWriter:
    def __init__(
            self,
            out_dir: str,
            participant_id: int,
            session_id: int,
    ) -> None:
        self._out_dir = Path(out_dir) / f'session{session_id}' / f'participant{participant_id}'
        self._out_dir.mkdir(parents=True, exist_ok=True)

        self._session_id = session_id
        self._participant_id = participant_id

    def save(
            self,
            array: np.ndarray,
            gesture_id: int,
            trial_id: int,
    ) -> str:
        file_name= f'session{self._session_id}_participant{self._participant_id}_gesture{gesture_id}_trial{trial_id}'
        file_path= self._out_dir / f'{file_name}.parquet'

        ch_cols = [f"ch{i}" for i in range(array.shape[1])]
        df_signal = pd.DataFrame(array, columns=ch_cols)

        df_meta = pd.DataFrame({
            "participant": self._participant_id,
            "session": self._session_id,
            "gesture": gesture_id,
            "trial": trial_id,
        }, index=range(len(array)))

        df = pd.concat([df_meta, df_signal], axis=1)
        df.to_parquet(file_path, index=False)

        return str(file_path)
         