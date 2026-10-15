"""Data access helpers shared by every module."""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from functools import lru_cache

import pandas as pd

from .config import DATA_DIR
from .reference import CITY_ALIASES, CITY_TO_STATIONS, STATION_INDEX

_STR_COLS = {"train_no": str, "pnr": str, "run_days": str}


@lru_cache(maxsize=None)
def _load(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        from .datagen import generate_all
        generate_all(verbose=False)
    return pd.read_csv(path, dtype=_STR_COLS)


def load(name: str) -> pd.DataFrame:
    """Return a copy so callers can mutate freely."""
    return _load(name).copy()


def clear_cache() -> None:
    _load.cache_clear()


def trains() -> pd.DataFrame:
    return _load("trains")


def train(train_no: str) -> dict:
    t = trains()
    row = t[t["train_no"] == str(train_no)]
    if row.empty:
        raise KeyError(f"Unknown train {train_no}")
    rec = row.iloc[0].to_dict()
    rec["capacity"] = json.loads(rec["capacity"])
    return rec


def resolve_city(text: str) -> str | None:
    """Map free text ('Bombay', 'NDLS', 'दिल्ली') to a canonical city name."""
    t = text.strip().lower()
    if t.upper() in STATION_INDEX:
        return STATION_INDEX[t.upper()][2].lower()
    t = CITY_ALIASES.get(t, t)
    return t if t in CITY_TO_STATIONS else None


def stations_for(city_or_code: str) -> list[str]:
    city = resolve_city(city_or_code)
    return CITY_TO_STATIONS.get(city, []) if city else []


def runs_on(run_days: str, d: date) -> bool:
    return run_days[d.weekday()] == "1"


def trains_between(src: str, dst: str, on: date | None = None) -> pd.DataFrame:
    s, d = stations_for(src), stations_for(dst)
    t = trains()
    out = t[t["src"].isin(s) & t["dst"].isin(d)].copy()
    if on is not None:
        out = out[out["run_days"].apply(lambda r: runs_on(r, on))]
    return out


def arrival_datetime(row, journey: date) -> datetime:
    dep = datetime.combine(journey, datetime.min.time()) + timedelta(minutes=int(row["dep_min"]))
    return dep + timedelta(minutes=int(row["duration_min"]))


def station_label(code: str) -> str:
    s = STATION_INDEX.get(code)
    return f"{s[1]} ({code})" if s else code
