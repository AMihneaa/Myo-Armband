import argparse
import asyncio
import csv
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

from myo import AggregatedData, MyoClient
from myo.types import (
    ClassifierEvent,
    ClassifierMode,
    EMGData,
    EMGDataSingle,
    EMGMode,
    FVData,
    IMUData,
    IMUMode,
    MotionEvent,
)

# ---------- config ----------
DEFAULT_MAC = "E4:96:A9:A7:5C:74"

PREP_SECONDS = 1.0
RECORD_SECONDS = 3.0

# 4 blocuri x (5 open + 5 fist) = 20 + 20
PROTOCOL = (
    ["open_hand_relaxed"] * 5
    + ["fist_closed"] * 5
    + ["open_hand_relaxed"] * 5
    + ["fist_closed"] * 5
    + ["open_hand_relaxed"] * 5
    + ["fist_closed"] * 5
    + ["open_hand_relaxed"] * 5
    + ["fist_closed"] * 5
)

CSV_COLUMNS = (
    ["timestamp_iso", "t_rel_s"]
    + [f"emg_{i}" for i in range(1, 9)]
    + [f"q{i}" for i in range(1, 5)]
    + [f"acc_{i}" for i in range(1, 4)]
    + [f"gyro_{i}" for i in range(1, 4)]
)


@dataclass
class TrialMeta:
    label: str
    trial_idx_for_label: int
    global_trial_idx: int


class TrialRecorder(MyoClient):
    def __init__(self, aggregate_all=False, aggregate_emg=False):
        super().__init__(aggregate_all=aggregate_all, aggregate_emg=aggregate_emg)
        self.recording = False
        self.rows = []
        self._trial_t0 = None

    async def on_classifier_event(self, ce: ClassifierEvent):
        # Nu avem nevoie de aceste date acum
        pass

    async def on_aggregated_data(self, ad: AggregatedData):
        if not self.recording:
            return

        values = self._parse_aggregated_data(ad)
        if values is None:
            return

        now_perf = time.perf_counter()
        row = {
            "timestamp_iso": datetime.now().isoformat(timespec="milliseconds"),
            "t_rel_s": now_perf - self._trial_t0,
        }

        # 8 EMG + 4 quat + 3 accel + 3 gyro = 18 valori
        for i in range(8):
            row[f"emg_{i+1}"] = values[i]
        for i in range(4):
            row[f"q{i+1}"] = values[8 + i]
        for i in range(3):
            row[f"acc_{i+1}"] = values[12 + i]
        for i in range(3):
            row[f"gyro_{i+1}"] = values[15 + i]

        self.rows.append(row)

    async def on_emg_data(self, emg: EMGData):
        pass

    async def on_emg_data_aggregated(self, emg: EMGDataSingle):
        # obligatoriu ca să nu rămână clasa abstractă
        pass

    async def on_fv_data(self, fvd: FVData):
        pass

    async def on_imu_data(self, imu: IMUData):
        pass

    async def on_motion_event(self, me: MotionEvent):
        pass

    def start_trial(self):
        self.rows = []
        self._trial_t0 = time.perf_counter()
        self.recording = True

    def stop_trial(self):
        self.recording = False

    @staticmethod
    def _parse_aggregated_data(ad: AggregatedData):
        """
        Din ce ai văzut în log, AggregatedData se stringify-uiește ca:
        emg1,...,emg8,q1,q2,q3,q4,acc1,acc2,acc3,gyro1,gyro2,gyro3
        """
        text = str(ad).strip()
        parts = [p.strip() for p in text.split(",")]

        if len(parts) < 18:
            logging.warning("AggregatedData în format neașteptat: %s", text)
            return None

        try:
            return [float(x) for x in parts[:18]]
        except ValueError:
            logging.warning("Nu am putut converti AggregatedData: %s", text)
            return None


def save_csv(rows, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def plot_trial(rows, emg_plot_path: Path, imu_plot_path: Path, title_prefix: str):
    if not rows:
        return

    t = [r["t_rel_s"] for r in rows]

    # -------- EMG figure --------
    fig, axes = plt.subplots(8, 1, figsize=(12, 14), sharex=True)
    for ch in range(8):
        y = [r[f"emg_{ch+1}"] for r in rows]
        axes[ch].plot(t, y)
        axes[ch].set_ylabel(f"EMG{ch+1}")
        axes[ch].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(f"{title_prefix} - EMG", fontsize=14)
    fig.tight_layout()
    emg_plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(emg_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # -------- IMU figure --------
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Quaternion
    for i in range(4):
        y = [r[f"q{i+1}"] for r in rows]
        axes[0].plot(t, y, label=f"q{i+1}")
    axes[0].set_ylabel("Quat")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", ncol=4)

    # Accelerometer
    for i in range(3):
        y = [r[f"acc_{i+1}"] for r in rows]
        axes[1].plot(t, y, label=f"acc_{i+1}")
    axes[1].set_ylabel("Accel")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right", ncol=3)

    # Gyroscope
    for i in range(3):
        y = [r[f"gyro_{i+1}"] for r in rows]
        axes[2].plot(t, y, label=f"gyro_{i+1}")
    axes[2].set_ylabel("Gyro")
    axes[2].set_xlabel("Time [s]")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right", ncol=3)

    fig.suptitle(f"{title_prefix} - IMU", fontsize=14)
    fig.tight_layout()
    imu_plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(imu_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_trial_paths(base_dir: Path, meta: TrialMeta):
    csv_path = base_dir / "raw" / meta.label / f"trial_{meta.trial_idx_for_label:02d}.csv"
    emg_plot_path = base_dir / "plots" / meta.label / f"trial_{meta.trial_idx_for_label:02d}_emg.png"
    imu_plot_path = base_dir / "plots" / meta.label / f"trial_{meta.trial_idx_for_label:02d}_imu.png"
    return csv_path, emg_plot_path, imu_plot_path


async def async_input(prompt: str) -> str:
    return await asyncio.to_thread(input, prompt)


async def run_protocol(recorder: TrialRecorder, base_dir: Path):
    label_counts = defaultdict(int)

    print("\nProtocol:")
    print("- open_hand_relaxed = brat intins, relaxat, mana deschisa")
    print("- fist_closed       = brat intins, pumn inchis")
    print(f"- pregatire: {PREP_SECONDS:.0f}s")
    print(f"- inregistrare: {RECORD_SECONDS:.0f}s")
    print("- apasa Enter pentru a porni fiecare proba")
    print("- scrie q si Enter daca vrei sa opresti\n")

    total_trials = len(PROTOCOL)

    for global_idx, label in enumerate(PROTOCOL, start=1):
        label_counts[label] += 1
        meta = TrialMeta(
            label=label,
            trial_idx_for_label=label_counts[label],
            global_trial_idx=global_idx,
        )

        prompt = (
            f"[{meta.global_trial_idx:02d}/{total_trials}] "
            f"{meta.label} | trial {meta.trial_idx_for_label:02d}/20 "
            f"-> apasa Enter pentru start (sau q + Enter pentru stop): "
        )
        user_cmd = (await async_input(prompt)).strip().lower()
        if user_cmd == "q":
            print("Sesiunea a fost oprita de utilizator.")
            break

        print(f"Pregateste-te... incepe in {PREP_SECONDS:.0f} secunda.")
        await asyncio.sleep(PREP_SECONDS)

        print("REC")
        recorder.start_trial()
        await asyncio.sleep(RECORD_SECONDS)
        recorder.stop_trial()
        print("STOP")

        if not recorder.rows:
            print("Nu s-au primit date pentru aceasta proba.\n")
            continue

        csv_path, emg_plot_path, imu_plot_path = build_trial_paths(base_dir, meta)
        save_csv(recorder.rows, csv_path)
        plot_trial(
            recorder.rows,
            emg_plot_path,
            imu_plot_path,
            title_prefix=f"{meta.label} | trial {meta.trial_idx_for_label:02d}",
        )

        print(f"Salvat CSV : {csv_path}")
        print(f"Salvat EMG : {emg_plot_path}")
        print(f"Salvat IMU : {imu_plot_path}\n")


async def main(args):
    base_dir = Path(args.out_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Connecting to Myo...")
    recorder = await TrialRecorder.with_device(mac=args.mac, aggregate_all=True)

    info = await recorder.get_services()
    logging.info("Services discovered successfully.")
    logging.debug(info)

    await recorder.setup(
        classifier_mode=ClassifierMode.ENABLED,
        emg_mode=EMGMode.SEND_FILT,
        imu_mode=IMUMode.SEND_ALL,
    )

    await recorder.start()
    print("\nMyo conectat. Stream-ul este activ.\n")

    try:
        await run_protocol(recorder, base_dir)
    finally:
        logging.info("Stopping notifications...")
        await recorder.stop()
        logging.info("Disconnecting...")
        await recorder.disconnect()
        print("Deconectat de la Myo.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mac", default=DEFAULT_MAC, help="MAC address-ul Myo")
    parser.add_argument(
        "--out-dir",
        default="data",
        help="directorul de iesire pentru csv-uri si ploturi",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="afiseaza loguri mai detaliate",
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)-15s %(name)-8s %(levelname)s: %(message)s",
    )

    asyncio.run(main(args))