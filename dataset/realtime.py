"""Replay the checked-in telemetry and camera datasets as realtime events.

The replay is deliberately file-backed: it has the same iterator interface a
live CSV/socket adapter can use later, while remaining deterministic for local
development and demos.
"""

from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class SensorFrame:
    """One telemetry sample from ``dataset/sensors.csv``."""

    sequence: int
    vehicle_type: str
    vehicle_age: int
    mileage: float
    engine_temp: float
    rpm: float
    oil_pressure: float
    fuel_consumption: float
    battery_voltage: float
    brake_pressure: float
    tire_pressure: float
    tire_condition: str
    vibration_level: str
    last_service_months: int
    accident_history: bool
    km_range_before_defect: str
    remaining_km: float

    @property
    def health_score(self) -> float:
        """Return a bounded 0..1 vehicle-health score from telemetry."""

        score = 1.0
        score -= max(0.0, self.engine_temp - 95.0) / 100.0
        score -= max(0.0, 30.0 - self.oil_pressure) / 100.0
        score -= max(0.0, 30.0 - self.tire_pressure) / 100.0
        score -= max(0.0, self.last_service_months - 6) / 60.0
        score -= 0.12 if self.accident_history else 0.0
        score -= {"Low": 0.0, "Medium": 0.08, "High": 0.18}.get(
            self.vibration_level, 0.0
        )
        return max(0.0, min(1.0, score))

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["health_score"] = round(self.health_score, 4)
        return payload


@dataclass(frozen=True)
class RealtimeEvent:
    """A synchronized telemetry and camera event."""

    sequence: int
    sensor: SensorFrame
    image_path: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "sensor": self.sensor.to_dict(),
            "image_path": self.image_path,
        }


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"yes", "true", "1"}


def iter_sensor_frames(csv_path: str | Path, limit: int | None = None) -> Iterator[SensorFrame]:
    """Yield validated telemetry rows in source order."""

    with Path(csv_path).open(newline="", encoding="utf-8") as source:
        for sequence, row in enumerate(csv.DictReader(source)):
            if limit is not None and sequence >= limit:
                break
            yield SensorFrame(
                sequence=sequence,
                vehicle_type=row["Vehicle_Type"],
                vehicle_age=int(row["Vehicle_Age"]),
                mileage=float(row["Mileage"]),
                engine_temp=float(row["Engine_Temp"]),
                rpm=float(row["RPM"]),
                oil_pressure=float(row["Oil_Pressure"]),
                fuel_consumption=float(row["Fuel_Consumption"]),
                battery_voltage=float(row["Battery_Voltage"]),
                brake_pressure=float(row["Brake_Pressure"]),
                tire_pressure=float(row["Tire_Pressure"]),
                tire_condition=row["Tire_Condition"],
                vibration_level=row["Vibration_Level"],
                last_service_months=int(row["Last_Service_Months"]),
                accident_history=_parse_bool(row["Accident_History"]),
                km_range_before_defect=row["KM_Range_Before_Defect"],
                remaining_km=float(row["Remaining_KM"]),
            )


def iter_image_frames(image_dir: str | Path, split: str = "val") -> Iterator[Path]:
    """Yield IDD camera images in deterministic path order."""

    root = Path(image_dir) / "leftImg8bit" / split
    yield from sorted(root.rglob("*_image.jpg"))


def stream_dataset(
    sensor_csv: str | Path,
    image_dir: str | Path | None = None,
    split: str = "val",
    limit: int | None = None,
    interval: float = 0.0,
) -> Iterator[RealtimeEvent]:
    """Replay telemetry and camera data as realtime events.

    Camera frames cycle when the telemetry file is longer than the image set.
    ``interval`` is the delay between events; keep it at zero for tests.
    """

    images = iter_image_frames(image_dir, split) if image_dir else iter(())
    image_paths = list(images)
    for sensor in iter_sensor_frames(sensor_csv, limit=limit):
        if interval > 0:
            time.sleep(interval)
        image_path = str(image_paths[sensor.sequence % len(image_paths)]) if image_paths else None
        yield RealtimeEvent(sensor.sequence, sensor, image_path)
