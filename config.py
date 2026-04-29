BLE_ADDR= "E4:96:A9:A7:5C:74"


SAMPLE_RATE= 200
IMU_RATE= 50
TRIAL_DURATION_S= 5.0

TRIAL_SAMPLES= int(SAMPLE_RATE * TRIAL_DURATION_S)
IMU_TRIAL_SAMPLES= int(IMU_RATE * TRIAL_DURATION_S)

N_CHANNELS= 8
N_GESTURES= 5
N_TRIALS= 7

PREPARE_DURATION_S= 3.0
REST_DURATION_S= 10.0


# EMG Pipeline
EMG_COLUMNS = [f"ch{i}" for i in range(8)]
BASE_STAT = [
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