"""Synthetic data generator for RailMind AI.

Produces realistic-looking (but entirely synthetic) Indian Railways datasets with
known causal structure, so every model has a learnable signal and the governance
layer has something meaningful to detect when we inject drift.

Run:  python -m railmind.datagen
"""
from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .config import DATA_DIR, SEED
from .reference import (CLASS_INFO, STATION_INDEX, STATIONS, TRAIN_TYPES, base_fare,
                        festival_intensity, fog_season, monsoon, rail_distance_km)

HIST_START = date(2024, 8, 1)
HIST_END = date(2026, 10, 14)

# (src, dst, [types]) — each corridor is generated in both directions
CORRIDORS = [
    ("NDLS", "CSMT", ["Rajdhani", "Superfast", "Mail"]),
    ("NZM", "MMCT", ["Rajdhani", "Express"]),
    ("NDLS", "HWH", ["Rajdhani", "Duronto", "Mail"]),
    ("NDLS", "SDAH", ["Rajdhani"]),
    ("NDLS", "MAS", ["Superfast", "Express"]),
    ("NZM", "SBC", ["Rajdhani", "Express"]),
    ("NZM", "SC", ["Rajdhani", "Express"]),
    ("PUNE", "NZM", ["Duronto", "Express"]),
    ("PUNE", "NDLS", ["Superfast"]),
    ("NDLS", "JP", ["Shatabdi", "Vande Bharat"]),
    ("NDLS", "LKO", ["Shatabdi", "Express"]),
    ("NDLS", "BSB", ["Vande Bharat", "Express"]),
    ("NDLS", "PNBE", ["Rajdhani", "Express"]),
    ("NDLS", "CDG", ["Shatabdi", "Vande Bharat"]),
    ("NDLS", "BPL", ["Shatabdi", "Express"]),
    ("NDLS", "GHY", ["Rajdhani"]),
    ("NDLS", "ADI", ["Rajdhani", "Mail"]),
    ("CSMT", "PUNE", ["Superfast", "Vande Bharat"]),
    ("MMCT", "ADI", ["Shatabdi", "Vande Bharat", "Express"]),
    ("CSMT", "HWH", ["Duronto", "Mail"]),
    ("CSMT", "MAS", ["Mail"]),
    ("CSMT", "SBC", ["Express"]),
    ("MMCT", "JP", ["Superfast"]),
    ("CSMT", "MAO", ["Vande Bharat", "Express"]),
    ("CSMT", "UD", ["Express"]),
    ("CSMT", "MAQ", ["Express"]),
    ("SBC", "MAS", ["Shatabdi", "Vande Bharat", "Express"]),
    ("SBC", "SC", ["Vande Bharat", "Express"]),
    ("SBC", "ERS", ["Express"]),
    ("SBC", "UD", ["Express"]),
    ("SBC", "MAQ", ["Express"]),
    ("MAS", "ERS", ["Mail"]),
    ("MAS", "HWH", ["Mail", "Superfast"]),
    ("MAS", "SC", ["Express"]),
    ("HWH", "PNBE", ["Vande Bharat", "Express"]),
    ("HWH", "BBS", ["Vande Bharat", "Express"]),
    ("HWH", "GHY", ["Superfast"]),
    ("HWH", "BSB", ["Express"]),
    ("PUNE", "HWH", ["Duronto", "Superfast"]),
    ("PUNE", "SC", ["Superfast"]),
    ("PUNE", "SBC", ["Express"]),
    ("PUNE", "ADI", ["Express"]),
    ("PUNE", "NGP", ["Superfast"]),
    ("NGP", "NDLS", ["Express"]),
    ("NGP", "HWH", ["Express"]),
    ("LKO", "PNBE", ["Express"]),
    ("LKO", "BSB", ["Express"]),
    ("LKO", "CSMT", ["Superfast"]),
    ("MAQ", "ERS", ["Express"]),
    ("MAO", "MAQ", ["Express"]),
    ("MAO", "UD", ["Express"]),
    ("ADI", "JP", ["Superfast"]),
    ("BPL", "PUNE", ["Express"]),
]

# Departure times we pin so the flagship demo query has meaningful answers
PINNED_DEPARTURES = {
    ("PUNE", "NZM", "Duronto"): "15:15",
    ("PUNE", "NZM", "Express"): "03:15",
    ("PUNE", "NDLS", "Superfast"): "08:40",
}

REGIONS = {"NR": "North", "NWR": "North", "NCR": "North", "CR": "West", "WR": "West", "KR": "West",
           "ER": "East", "ECR": "East", "ECoR": "East", "SR": "South", "SWR": "South",
           "SCR": "South", "WCR": "Central", "NFR": "North-East"}


def _hhmm(minutes: float) -> str:
    m = int(minutes) % (24 * 60)
    return f"{m // 60:02d}:{m % 60:02d}"


def make_trains(rng: np.random.Generator) -> pd.DataFrame:
    rows, num = [], 12001
    for src, dst, types in CORRIDORS:
        dist = rail_distance_km(src, dst)
        for ttype in types:
            speed, classes, coaches, punct, _ = TRAIN_TYPES[ttype]
            if ttype in ("Vande Bharat", "Shatabdi") and dist > 1000:
                ttype, (speed, classes, coaches, punct, _) = "Superfast", TRAIN_TYPES["Superfast"]
            for a, b in ((src, dst), (dst, src)):
                dur_min = dist / (speed * rng.uniform(0.93, 1.07)) * 60
                pinned = PINNED_DEPARTURES.get((a, b, ttype))
                if pinned:
                    h, m = map(int, pinned.split(":"))
                    dep = h * 60 + m
                elif ttype in ("Vande Bharat", "Shatabdi"):
                    dep = rng.choice([6 * 60, 6 * 60 + 10, 6 * 60 + 40, 14 * 60 + 50, 15 * 60 + 30])
                elif dist > 900:
                    dep = int(rng.choice([rng.integers(15 * 60, 23 * 60), rng.integers(5 * 60, 12 * 60)]))
                else:
                    dep = int(rng.integers(18 * 60, 23 * 60 + 30))
                arr = dep + dur_min
                if pinned:
                    run_days = "1111111"
                else:
                    run_days = "1111111" if rng.random() < 0.65 or ttype == "Vande Bharat" else \
                        "".join("1" if rng.random() < 0.55 else "0" for _ in range(7))
                    if "1" not in run_days:
                        run_days = "1010101"
                city_a, city_b = STATION_INDEX[a][2], STATION_INDEX[b][2]
                rows.append({
                    "train_no": str(num),
                    "train_name": f"{city_a}–{city_b} {ttype}",
                    "train_type": ttype,
                    "src": a, "dst": b,
                    "dep_time": _hhmm(dep), "arr_time": _hhmm(arr),
                    "dep_min": int(dep) % 1440,
                    "duration_min": int(round(dur_min)),
                    "day_offset": int((dep + dur_min) // 1440),
                    "distance_km": dist,
                    "classes": ",".join(classes),
                    "capacity": json.dumps({c: coaches[c] * CLASS_INFO[c][2] for c in classes}),
                    "run_days": run_days,
                    "punctuality": round(float(np.clip(punct + rng.normal(0, 0.05), 0.4, 0.98)), 3),
                    "zone": STATION_INDEX[a][5],
                    "cancel_rate_base": round(float(rng.uniform(0.10, 0.24)), 3),
                })
                num += 2 if rng.random() < 0.5 else 3
    df = pd.DataFrame(rows)
    df["train_no"] = df["train_no"].astype(str)
    return df


def _rand_dates(rng, n, start=HIST_START, end=HIST_END):
    span = (end - start).days
    return [start + timedelta(days=int(x)) for x in rng.integers(0, span, n)]


def make_bookings(trains: pd.DataFrame, rng, n=60000) -> pd.DataFrame:
    """Waiting-list / RAC bookings with their final outcome after chart preparation."""
    t = trains.sample(n, replace=True, random_state=SEED).reset_index(drop=True)
    jdates = _rand_dates(rng, n)
    out = []
    for i, row in t.iterrows():
        classes = row["classes"].split(",")
        cls = classes[rng.integers(0, len(classes))]
        cap = json.loads(row["capacity"])[cls]
        jd = jdates[i]
        fest = festival_intensity(jd)
        lead = int(np.clip(rng.gamma(2.0, 12.0), 0, 60))
        quota = "TQ" if lead <= 1 and rng.random() < 0.6 else ("LD" if rng.random() < 0.05 else "GN")
        demand = 0.55 + 0.6 * fest + (0.1 if jd.weekday() in (4, 5, 6) else 0) + rng.normal(0, 0.08)
        is_rac = rng.random() < 0.18 and cls not in ("2S", "CC", "EC", "1A")
        rac_cap = int(cap * 0.06) if cls in ("SL", "3A", "3E", "2A") else 0
        if is_rac and rac_cap > 0:
            status0, pos = "RAC", int(rng.integers(1, rac_cap + 1))
        else:
            status0 = "WL"
            pos = int(np.clip(rng.gamma(1.6, 18 + 70 * max(demand - 0.5, 0)), 1, 400))
        cancel_rate = row["cancel_rate_base"] * (1 - 0.45 * fest) * (0.75 if quota == "TQ" else 1.0)
        lead_factor = 1 - np.exp(-(lead + 1) / 14)
        expected_cancel = cap * cancel_rate * (0.35 + 0.65 * lead_factor)
        cancels = rng.poisson(max(expected_cancel, 0.1))
        # RAC passengers ahead of the WL queue absorb cancellations first
        effective = pos * 2.2 if status0 == "RAC" else pos + rac_cap
        if effective <= cancels:
            final = "CNF"
        elif status0 == "WL" and pos <= cancels and rac_cap > 0:
            final = "RAC"
        elif status0 == "RAC":
            final = "RAC"
        else:
            final = "WL"
        out.append({
            "train_no": row["train_no"], "train_type": row["train_type"], "class": cls,
            "journey_date": jd.isoformat(), "booking_lead_days": lead, "quota": quota,
            "initial_status": status0, "initial_position": pos, "class_capacity": cap,
            "distance_km": row["distance_km"], "month": jd.month, "day_of_week": jd.weekday(),
            "festival_intensity": fest, "route_demand_index": round(float(demand), 3),
            "hist_cancel_rate": round(float(row["cancel_rate_base"]), 3),
            "final_status": final, "confirmed": int(final == "CNF"),
        })
    return pd.DataFrame(out)


def _weather(rng, d: date, zone: str) -> str:
    north = zone in ("NR", "NWR", "NCR", "ECR", "NFR")
    p = rng.random()
    if fog_season(d) and north and p < 0.45:
        return "fog"
    if monsoon(d) and p < (0.55 if zone in ("KR", "SR", "CR", "WR", "NFR") else 0.40):
        return "storm" if rng.random() < 0.15 else "rain"
    return "clear"


def delay_minutes(rng, ttype, dist, weather, fest, cong, prior, block, punct, hour):
    base = {"Vande Bharat": 6, "Rajdhani": 12, "Shatabdi": 9, "Duronto": 18,
            "Superfast": 28, "Express": 38, "Mail": 45}[ttype]
    w = {"clear": 0, "rain": 22, "storm": 65, "fog": 95}[weather] * (0.5 + dist / 2000)
    d = (base + 0.012 * dist + w + 35 * fest + 30 * cong + 0.75 * prior + 55 * block
         + 60 * (1 - punct) + (8 if 6 <= hour <= 10 else 0))
    d *= rng.lognormal(0, 0.35)
    return float(0.55 * max(d - 15, 0) + rng.normal(0, 3))


def make_delays(trains: pd.DataFrame, rng, n=50000) -> pd.DataFrame:
    t = trains.sample(n, replace=True, random_state=SEED + 1).reset_index(drop=True)
    dates = _rand_dates(rng, n)
    rows = []
    for i, row in t.iterrows():
        d = dates[i]
        zone = STATION_INDEX[row["dst"]][5]
        weather = _weather(rng, d, zone)
        fest = festival_intensity(d)
        arr_hour = int(row["arr_time"][:2])
        cong = float(np.clip(STATION_INDEX[row["dst"]][6] * (0.6 + 0.4 * fest)
                             + (0.15 if 7 <= arr_hour <= 11 or 17 <= arr_hour <= 21 else 0)
                             + rng.normal(0, 0.07), 0, 1))
        prior = max(rng.gamma(1.2, 6 + 25 * (weather == "fog")) - 4, 0)
        block = int(rng.random() < 0.06)
        delay = max(delay_minutes(rng, row["train_type"], row["distance_km"], weather, fest, cong,
                                  prior, block, row["punctuality"], arr_hour), 0)
        rows.append({
            "train_no": row["train_no"], "train_type": row["train_type"], "src": row["src"],
            "dst": row["dst"], "journey_date": d.isoformat(), "month": d.month,
            "day_of_week": d.weekday(), "sched_arr_hour": arr_hour,
            "distance_km": row["distance_km"], "zone": zone, "weather": weather,
            "festival_intensity": fest, "station_congestion": round(cong, 3),
            "departure_delay_min": round(prior, 1), "maintenance_block": block,
            "punctuality_score": row["punctuality"], "arrival_delay_min": round(delay, 1),
        })
    return pd.DataFrame(rows)


def make_station_load(rng, days=420) -> pd.DataFrame:
    rows = []
    start = HIST_END - timedelta(days=days)
    for code, _, _, _, _, _, base in STATIONS:
        for d in (start + timedelta(days=i) for i in range(days)):
            fest = festival_intensity(d)
            for hour in (6, 9, 12, 15, 18, 21):
                peak = 1.0 if hour in (9, 18) else (0.8 if hour in (6, 21) else 0.6)
                load = 100 * base * peak * (0.75 + 0.45 * fest + (0.08 if d.weekday() >= 4 else 0))
                load += rng.normal(0, 5)
                rows.append({"station": code, "date": d.isoformat(), "hour": hour,
                             "day_of_week": d.weekday(), "month": d.month,
                             "festival_intensity": fest, "footfall_index": round(float(np.clip(load, 0, 150)), 1)})
    return pd.DataFrame(rows)


def make_fares(trains: pd.DataFrame, rng, n=40000) -> pd.DataFrame:
    t = trains.sample(n, replace=True, random_state=SEED + 2).reset_index(drop=True)
    dates = _rand_dates(rng, n)
    rows = []
    for i, row in t.iterrows():
        classes = row["classes"].split(",")
        cls = classes[rng.integers(0, len(classes))]
        jd = dates[i]
        fest = festival_intensity(jd)
        lead = int(rng.integers(0, 61))
        quota = "TQ" if lead <= 1 and cls not in ("1A", "EC") and rng.random() < 0.7 else "GN"
        # occupancy rises as journey approaches, faster in festivals
        occ = float(np.clip(1 - np.exp(-(60 - lead) / (22 - 10 * fest)) + rng.normal(0, 0.05) + 0.15 * fest, 0, 1))
        base = base_fare(cls, row["distance_km"], row["train_type"])
        flexi = TRAIN_TYPES[row["train_type"]][4] and cls != "1A"
        mult = 1.0
        if flexi:
            mult = 1 + min(0.5, 0.1 * max(0, int(occ * 10) - 1))
        if quota == "TQ":
            surcharge = {"2S": 15, "SL": 150, "CC": 190, "3E": 350, "3A": 380, "2A": 470, "EC": 470}.get(cls, 0)
            mult = 1.0 + (0.3 if row["train_type"] in ("Superfast", "Express", "Mail") and occ > 0.85 else 0)
            fare = base * mult + surcharge
        else:
            fare = base * mult
        fare = round(fare + rng.normal(0, 3), 0)
        rows.append({"train_no": row["train_no"], "train_type": row["train_type"], "class": cls,
                     "distance_km": row["distance_km"], "journey_date": jd.isoformat(),
                     "days_before": lead, "quota": quota, "occupancy_pct": round(occ * 100, 1),
                     "festival_intensity": fest, "day_of_week": jd.weekday(), "month": jd.month,
                     "base_fare": base, "fare": float(fare)})
    return pd.DataFrame(rows)


def make_demand(trains: pd.DataFrame, rng) -> pd.DataFrame:
    corr = trains[["src", "dst"]].drop_duplicates().values.tolist()
    rows = []
    days = (HIST_END - HIST_START).days
    for src, dst in corr:
        scale = rng.uniform(300, 2500)
        for i in range(days):
            d = HIST_START + timedelta(days=i)
            fest = festival_intensity(d)
            weekly = 1 + 0.18 * (d.weekday() in (4, 6)) + 0.08 * (d.weekday() == 5)
            trend = 1 + 0.0002 * i
            val = scale * weekly * trend * (1 + 0.9 * fest) * rng.lognormal(0, 0.08)
            rows.append({"src": src, "dst": dst, "date": d.isoformat(), "bookings": int(val),
                         "day_of_week": d.weekday(), "month": d.month, "festival_intensity": fest})
    return pd.DataFrame(rows)


def make_users(rng, n=5000) -> pd.DataFrame:
    cities = sorted({s[2] for s in STATIONS})
    rows = []
    for i in range(n):
        budget = rng.choice(["low", "mid", "high"], p=[0.45, 0.4, 0.15])
        pref = {"low": ["SL", "2S", "3E"], "mid": ["3A", "CC", "3E"], "high": ["2A", "1A", "EC"]}[budget]
        rows.append({
            "user_id": f"U{i:05d}",
            "gender": rng.choice(["F", "M"], p=[0.42, 0.58]),
            "age_group": rng.choice(["18-25", "26-40", "41-60", "60+"], p=[0.3, 0.4, 0.22, 0.08]),
            "home_city": rng.choice(cities),
            "region": rng.choice(["North", "South", "East", "West", "North-East", "Central"],
                                 p=[0.27, 0.22, 0.2, 0.2, 0.05, 0.06]),
            "budget_level": budget,
            "preferred_class": rng.choice(pref),
            "preferred_time": rng.choice(["morning", "afternoon", "night"], p=[0.35, 0.2, 0.45]),
            "punctuality_weight": round(float(rng.uniform(0.2, 1.0)), 2),
            "past_trips": int(rng.poisson(6)),
        })
    return pd.DataFrame(rows)


def time_bucket(dep_min: int) -> str:
    h = dep_min // 60
    if 5 <= h < 12:
        return "morning"
    if 12 <= h < 18:
        return "afternoon"
    return "night"


def make_choices(trains: pd.DataFrame, users: pd.DataFrame, rng, sessions=9000) -> pd.DataFrame:
    """Historical recommendation sessions: each user saw candidates on a corridor and chose one."""
    groups = {k: g for k, g in trains.groupby(["src", "dst"])}
    multi = [k for k, g in groups.items() if len(g) >= 2]
    budget_cap = {"low": 1.0, "mid": 2.2, "high": 4.0}  # INR per km tolerance proxy
    rows = []
    for s in range(sessions):
        u = users.iloc[int(rng.integers(0, len(users)))]
        key = multi[int(rng.integers(0, len(multi)))]
        cands = groups[key]
        options = []
        for _, tr in cands.iterrows():
            for cls in tr["classes"].split(","):
                options.append((tr, cls))
        if len(options) < 3:
            continue
        idx = rng.choice(len(options), size=min(len(options), 6), replace=False)
        feats, utils = [], []
        for j in idx:
            tr, cls = options[j]
            fare = base_fare(cls, tr["distance_km"], tr["train_type"])
            avail = float(np.clip(rng.beta(4, 2) - 0.25 * festival_intensity(date.today()), 0.05, 1))
            per_km = fare / tr["distance_km"]
            over = max(0.0, per_km - budget_cap[u["budget_level"]])
            util = (-2.2 * over - 0.35 * tr["duration_min"] / 600 + 1.4 * u["punctuality_weight"] * tr["punctuality"]
                    + 0.9 * (cls == u["preferred_class"]) + 0.6 * (time_bucket(tr["dep_min"]) == u["preferred_time"])
                    + 1.0 * avail + rng.gumbel(0, 0.6))
            feats.append({"session_id": s, "user_id": u["user_id"], "train_no": tr["train_no"],
                          "class": cls, "fare": fare, "duration_min": tr["duration_min"],
                          "punctuality": tr["punctuality"], "dep_min": tr["dep_min"],
                          "distance_km": tr["distance_km"], "train_type": tr["train_type"],
                          "availability": round(avail, 3)})
            utils.append(util)
        best = int(np.argmax(utils))
        for k, f in enumerate(feats):
            f["chosen"] = int(k == best)
            rows.append(f)
    return pd.DataFrame(rows)


def make_sessions(rng, n=25000) -> pd.DataFrame:
    """Booking sessions for fraud / bot detection with labelled attack types."""
    rows = []
    for i in range(n):
        r = rng.random()
        kind = "legit"
        if r < 0.022:
            kind = "bulk_bot"
        elif r < 0.034:
            kind = "card_testing"
        elif r < 0.045:
            kind = "account_takeover"
        elif r < 0.055:
            kind = "agent_misuse"
        hour = int(rng.integers(0, 24))
        rec = {
            "session_id": f"S{i:06d}", "account_age_days": float(rng.gamma(2, 300)),
            "bookings_last_24h": int(rng.poisson(0.6)), "passengers_per_booking": int(rng.integers(1, 5)),
            "session_duration_s": float(rng.gamma(3, 90)), "avg_click_interval_ms": float(rng.normal(1800, 500)),
            "form_fill_time_s": float(rng.gamma(4, 12)), "captcha_failures": int(rng.poisson(0.2)),
            "payment_failures": int(rng.poisson(0.1)), "distinct_cards_7d": int(1 + rng.poisson(0.2)),
            "accounts_per_device": int(1 + rng.poisson(0.1)), "ip_reputation": float(rng.beta(8, 2)),
            "is_tatkal": int(rng.random() < 0.2), "booking_hour": hour,
            "geo_mismatch": int(rng.random() < 0.03), "amount": float(rng.gamma(2, 900)),
        }
        if kind == "bulk_bot":
            rec.update(bookings_last_24h=int(rng.integers(6, 40)), avg_click_interval_ms=float(rng.normal(180, 60)),
                       form_fill_time_s=float(rng.uniform(1, 6)), is_tatkal=1, booking_hour=int(rng.choice([10, 11])),
                       passengers_per_booking=int(rng.integers(4, 7)), ip_reputation=float(rng.beta(2, 5)),
                       captcha_failures=int(rng.poisson(2)))
        elif kind == "card_testing":
            rec.update(payment_failures=int(rng.integers(3, 12)), distinct_cards_7d=int(rng.integers(4, 15)),
                       amount=float(rng.uniform(50, 400)), account_age_days=float(rng.uniform(0, 20)))
        elif kind == "account_takeover":
            rec.update(geo_mismatch=1, ip_reputation=float(rng.beta(2, 4)), booking_hour=int(rng.choice([1, 2, 3, 4])),
                       amount=float(rng.gamma(4, 1500)), session_duration_s=float(rng.uniform(20, 90)))
        elif kind == "agent_misuse":
            rec.update(accounts_per_device=int(rng.integers(4, 20)), bookings_last_24h=int(rng.integers(4, 15)),
                       is_tatkal=1, form_fill_time_s=float(rng.uniform(3, 12)))
        if kind == "legit" and rng.random() < 0.04:   # legitimate power users / families / autofill
            rec.update(bookings_last_24h=int(rng.integers(2, 7)), avg_click_interval_ms=float(rng.normal(450, 150)),
                       form_fill_time_s=float(rng.uniform(4, 15)), passengers_per_booking=int(rng.integers(4, 7)))
        if kind != "legit" and rng.random() < 0.35:   # stealthy attackers mimic humans on half the signals
            legit_like = {"avg_click_interval_ms": float(rng.normal(1500, 500)), "form_fill_time_s": float(rng.gamma(4, 10)),
                          "ip_reputation": float(rng.beta(6, 3)), "captcha_failures": 0, "geo_mismatch": 0,
                          "bookings_last_24h": int(rng.integers(1, 4)), "accounts_per_device": int(rng.integers(1, 3)),
                          "payment_failures": int(rng.integers(0, 3)), "distinct_cards_7d": int(rng.integers(1, 4))}
            for key in rng.choice(list(legit_like), size=5, replace=False):
                rec[key] = legit_like[key]
        rec["avg_click_interval_ms"] = max(rec["avg_click_interval_ms"], 40.0)
        rec["fraud_type"] = kind
        rec["is_fraud"] = int(kind != "legit")
        rows.append(rec)
    return pd.DataFrame(rows)


def make_pnrs(trains: pd.DataFrame, rng, n=400) -> pd.DataFrame:
    rows = []
    today = HIST_END
    for i in range(n):
        tr = trains.iloc[int(rng.integers(0, len(trains)))]
        cls = tr["classes"].split(",")[0]
        jd = today + timedelta(days=int(rng.integers(-20, 45)))
        status = rng.choice(["CNF", "RAC", "WL", "CAN"], p=[0.6, 0.12, 0.18, 0.10])
        pax = int(rng.integers(1, 5))
        fare = base_fare(cls, tr["distance_km"], tr["train_type"]) * pax
        detail = {"CNF": f"CNF/{rng.choice(['B','S','A'])}{rng.integers(1,9)}/{rng.integers(1,72)}",
                  "RAC": f"RAC {rng.integers(1,30)}", "WL": f"GNWL {rng.integers(1,120)}", "CAN": "Cancelled"}[status]
        refund = {"CAN": rng.choice(["Refund processed", "Refund initiated", "Refund under process"])}.get(status, "-")
        rows.append({"pnr": f"{int(rng.integers(2_100_000_000, 5_999_999_999))}", "train_no": tr["train_no"],
                     "train_name": tr["train_name"], "src": tr["src"], "dst": tr["dst"], "class": cls,
                     "journey_date": jd.isoformat(), "passengers": pax, "status": status,
                     "status_detail": detail, "amount": float(fare), "refund_status": refund})
    return pd.DataFrame(rows)


def generate_all(verbose: bool = True) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    trains = make_trains(rng)
    users = make_users(rng)
    data = {
        "trains": trains,
        "bookings": make_bookings(trains, rng),
        "delays": make_delays(trains, rng),
        "station_load": make_station_load(rng),
        "fares": make_fares(trains, rng),
        "demand": make_demand(trains, rng),
        "users": users,
        "choices": make_choices(trains, users, rng),
        "sessions": make_sessions(rng),
        "pnrs": make_pnrs(trains, rng),
    }
    for name, df in data.items():
        df.to_csv(DATA_DIR / f"{name}.csv", index=False)
        if verbose:
            print(f"  {name:<13} {len(df):>7,} rows")
    return data


if __name__ == "__main__":
    print(f"Generating synthetic data into {DATA_DIR}")
    generate_all()
