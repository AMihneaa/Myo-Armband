import pandas as pd
import numpy as np

from pathlib import Path
from emg.processor import EMGProcessor

from config import EMG_COLUMNS


def load_myo_parquet(
            parquet_path: str | Path,
            processor: EMGProcessor,
            fs: float= 200.0,
        ):
    if not Path(parquet_path).exists():
        raise FileNotFoundError(f'File not found! :{parquet_path}')
    
    if not isinstance(processor, EMGProcessor):
        raise TypeError(f'processor must be an EMGProcessor instance, got {type(processor).__name__}')
    
    if not fs > 0:
        raise ValueError(f'Sampling Rate error: {fs}')

    df= pd.read_parquet(parquet_path)
    
    missing= [c for c in EMG_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f'Missing EMG channel columns: {missing}')

    signal= df[EMG_COLUMNS].to_numpy(dtype= np.float32)

    return processor.process_array(signal, fs_original= fs)

    
