"""Model health monitoring: SLA tracking, service health and resource utilisation."""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from ..config import SLA
from .store import Store

MODELS = ["confirmation_predictor", "train_recommender", "delay_predictor", "travel_planner_agent",
          "fare_predictor", "support_assistant", "fraud_detector"]


def window_start(minutes: int = 60, since: float | None = None) -> float:
    start = time.time() - minutes * 60
    return max(start, since or 0)


def health(model: str, minutes: int = 60, since: float | None = None) -> dict:
    df = Store.get().inferences(model, window_start(minutes, since))
    if df.empty:
        return {"model": model, "requests": 0, "rpm": 0.0, "status": "idle", "p50_ms": None, "p95_ms": None,
                "error_rate": 0.0, "blocked_rate": 0.0, "avg_dq": None, "availability": 1.0, "sla_ok": True,
                "breaches": [], "versions": []}
    ok = df[df["status"] == "ok"]
    p95 = float(np.percentile(ok["latency_ms"], 95)) if len(ok) else None
    err = float((df["status"] == "error").mean())
    blocked = float((df["status"] == "blocked").mean())
    avail = 1 - err
    breaches = []
    if p95 is not None and p95 > SLA["p95_latency_ms"]:
        breaches.append(f"p95 latency {p95:.0f} ms > {SLA['p95_latency_ms']:.0f} ms")
    if err > SLA["error_rate"]:
        breaches.append(f"error rate {err:.1%} > {SLA['error_rate']:.0%}")
    if avail < SLA["availability"]:
        breaches.append(f"availability {avail:.2%} < {SLA['availability']:.1%}")
    status = "healthy" if not breaches and blocked < 0.05 else ("degraded" if avail > 0.9 else "down")
    return {"model": model, "requests": int(len(df)), "rpm": round(len(df) / max(minutes, 1), 2),
            "p50_ms": round(float(np.percentile(ok["latency_ms"], 50)), 1) if len(ok) else None,
            "p95_ms": round(p95, 1) if p95 is not None else None, "error_rate": round(err, 4),
            "blocked_rate": round(blocked, 4), "avg_dq": round(float(df["dq_score"].mean()), 3),
            "availability": round(avail, 4), "sla_ok": not breaches, "breaches": breaches, "status": status,
            "versions": sorted(df["version"].dropna().unique().tolist())}


def health_table(minutes: int = 60) -> pd.DataFrame:
    return pd.DataFrame([health(m, minutes) for m in MODELS])


def recent_inputs(model: str, minutes: int = 60, since: float | None = None, status: str | None = "ok") -> pd.DataFrame:
    df = Store.get().inferences(model, window_start(minutes, since))
    if status:
        df = df[df["status"] == status]
    if df.empty:
        return pd.DataFrame()
    inputs = pd.DataFrame([json.loads(x) for x in df["input"]])
    inputs["_prediction"] = df["prediction"].values
    inputs["_label"] = df["label"].values
    inputs["_segment"] = df["segment"].values
    inputs["_epoch"] = df["epoch"].values
    inputs["_version"] = df["version"].values
    return inputs


def timeseries(model: str, hours: int = 24, bucket_min: int = 30) -> pd.DataFrame:
    df = Store.get().inferences(model, time.time() - hours * 3600, limit=200000)
    if df.empty:
        return df
    df["bucket"] = pd.to_datetime((df["epoch"] // (bucket_min * 60)) * bucket_min * 60, unit="s")
    g = df.groupby("bucket")
    out = pd.DataFrame({"requests": g.size(), "p95_ms": g["latency_ms"].quantile(0.95),
                        "error_rate": g["status"].apply(lambda s: (s == "error").mean()),
                        "blocked_rate": g["status"].apply(lambda s: (s == "blocked").mean()),
                        "avg_dq": g["dq_score"].mean()})
    lab = df.dropna(subset=["label", "prediction"])
    if not lab.empty:
        lab = lab.assign(abs_err=(lab["label"] - lab["prediction"]).abs())
        out["mean_abs_error"] = lab.groupby("bucket")["abs_err"].mean()
    return out.reset_index()


def resource_utilization() -> dict:
    try:
        import psutil  # optional
        vm = psutil.virtual_memory()
        return {"cpu_pct": psutil.cpu_percent(interval=0.2), "mem_pct": vm.percent,
                "mem_used_gb": round(vm.used / 1e9, 2), "source": "psutil"}
    except ImportError:
        load = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0
        cpus = os.cpu_count() or 1
        mem_pct = None
        try:
            with open("/proc/meminfo") as f:
                info = {l.split(":")[0]: int(l.split()[1]) for l in f}
            mem_pct = round(100 * (1 - info["MemAvailable"] / info["MemTotal"]), 1)
        except (OSError, KeyError, ValueError):
            pass
        return {"cpu_pct": round(min(100.0, 100 * load / cpus), 1), "mem_pct": mem_pct, "source": "loadavg"}
