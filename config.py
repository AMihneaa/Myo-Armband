BLE_ADDR= "E4:96:A9:A7:5C:74"

SAMPLE_RATE: int= 200
IMU_RATE: int= 50
N_CHANNELS: int= 8

TRIAL_DURATION_S: float= 5.0
TRIAL_SAMPLES: int= int(SAMPLE_RATE * TRIAL_DURATION_S)
IMU_TRIAL_SAMPLES: int= int(IMU_RATE * TRIAL_DURATION_S)

PREPARE_DURATION_S: float= 3.0
REST_BETWEEN_TRIALS_S: float= 10.0
REST_BETWEEN_GESTURES_S: float= 30.0

FAMILIARIZATION_REPS: int= 2
FAMILIARIZATION_DURATION_S: float= 5.0
FAMILIARIZATION_REST_S: float= REST_BETWEEN_TRIALS_S / 2

N_TRIALS: int= 7

GESTURE_LABELS: dict[int, str]= {
    0: "relax",
    1: "hand_open",
    2: "hand_close",
    3: "wrist_extension",
    4: "wrist_flexion",
}

HDF5_DIR: str= "data_raw"
HDF5_FILENAME_TEMPLATE: str= "subject_{subject_id:02d}_session_{session_id}.h5"

BP_LOW: float= 10.0
BP_HIGH: float= 95.0
ENV_CUTOFF: float= 10.0
NOTCH_FREQ: float= 50.0
NOTCH_Q: float= 30.0
APPLY_NOTCH: bool= True


BASE_STAT= [
    "env_rms",
    "env_mav",
    "raw_rms",
    "raw_mav",
    "iemg",
    "ssi",
    "var",
    "wl",
    "zc",
    "ssc",
    "wamp",
    "log_rms",
    "mnf",
    "mdf",
]

API_WS_HOST= "localhost"
API_WS_PORT= 8000