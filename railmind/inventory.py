"""Simulated seat inventory and booking ledger (stand-in for the PRS/IRCTC reservation system)."""
from __future__ import annotations

import hashlib
import random
import uuid
from datetime import date

from . import data
from .governance.store import Store
from .reference import base_fare, festival_intensity

_BOOKINGS: dict[str, dict] = {}


def availability(train_no: str, cls: str, journey: date, today: date | None = None) -> dict:
    """Deterministic pseudo-availability so the same query always returns the same answer."""
    today = today or date.today()
    t = data.train(train_no)
    cap = t["capacity"].get(cls, 0)
    seed = int(hashlib.md5(f"{train_no}{cls}{journey}".encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    lead = max((journey - today).days, 0)
    fest = festival_intensity(journey)
    pressure = 0.85 * (1 - min(lead, 60) / 60) + 0.3 * fest + (0.08 if journey.weekday() in (4, 6) else 0) + rng.uniform(-0.3, 0.3)
    pressure += {"SL": 0.15, "3A": 0.1, "3E": 0.05, "2A": -0.05, "1A": -0.15, "CC": 0.0, "EC": -0.1, "2S": 0.2}.get(cls, 0)
    if pressure < 0.68:
        avail = max(int(cap * (0.68 - pressure) * rng.uniform(0.3, 0.9)), 1)
        return {"status": "AVAILABLE", "count": avail, "label": f"AVL {avail}"}
    if pressure < 0.8 and cls in ("SL", "3A", "3E", "2A"):
        pos = rng.randint(1, max(int(cap * 0.06), 2))
        return {"status": "RAC", "count": pos, "label": f"RAC {pos}"}
    pos = int(1 + max(pressure - 0.8, 0) * cap * 0.12 + rng.randint(0, 15))
    return {"status": "WL", "count": max(pos, 1), "label": f"GNWL {max(pos, 1)}"}


def fare(train_no: str, cls: str) -> int:
    t = data.train(train_no)
    return base_fare(cls, t["distance_km"], t["train_type"])


def book(train_no: str, cls: str, journey: date, passengers: int, approved_by: str, approval_token: str | None,
         channel: str = "travel_planner_agent") -> dict:
    """Booking requires an explicit approval token — agents can never book on their own."""
    store = Store.get()
    if not approval_token:
        store.audit(channel, "booking.denied", train_no, {"reason": "missing user approval"}, "agent")
        raise PermissionError("Booking requires explicit passenger approval")
    av = availability(train_no, cls, journey)
    pnr = str(int(uuid.uuid4().int % 4_900_000_000) + 1_000_000_000)  # PNRs never look like mobile numbers
    rec = {"pnr": pnr, "train_no": train_no, "class": cls, "journey_date": journey.isoformat(),
           "passengers": passengers, "status": av["label"] if av["status"] != "AVAILABLE" else "CNF",
           "amount": fare(train_no, cls) * passengers, "approved_by": approved_by}
    _BOOKINGS[pnr] = rec
    store.audit(channel, "booking.created", train_no, {**rec, "approval_token": approval_token[:6] + "…"}, "agent")
    return rec


def get_booking(pnr: str) -> dict | None:
    if pnr in _BOOKINGS:
        return _BOOKINGS[pnr]
    p = data.load("pnrs")
    row = p[p["pnr"] == str(pnr)]
    return row.iloc[0].to_dict() if not row.empty else None
