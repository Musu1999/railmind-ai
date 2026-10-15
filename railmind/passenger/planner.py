"""4. AI Travel Planner Agent — multi-agent orchestration with governance built in.

    PlannerAgent (orchestrator)
      ├── IntentAgent        understands the natural-language request
      ├── SearchAgent        direct + one-change itineraries
      ├── AvailabilityAgent  seat inventory + WL confirmation predictor
      ├── DelayAgent         delay predictor -> risk-adjusted arrival time
      ├── FareAgent          fares (+ forecast of fare movement)
      ├── OptimizerAgent     constraint filtering + multi-objective ranking
      └── ValidatorAgent     governance check of the final plan before it is shown
    Booking happens only through `book()` with an explicit approval token from the passenger.
"""
from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

from .. import data, faults, inventory
from ..config import GOVERNANCE
from ..governance import registry
from ..governance.store import Store
from ..reference import CITY_ALIASES, CITY_TO_STATIONS, CLASS_INFO, STATION_INDEX

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov",
                                      "dec"], start=1)}
CLASS_WORDS = [
    (r"\b(sleeper|sl)\b", "SL"), (r"\b(3a|3ac|ac ?3|third ac|3 tier|3-tier)\b", "3A"),
    (r"\b(2a|2ac|ac ?2|second ac|2 tier|2-tier)\b", "2A"), (r"\b(1a|1ac|first ac|first class)\b", "1A"),
    (r"\b(chair car|cc)\b", "CC"), (r"\b(executive|ec)\b", "EC"), (r"\b(3e|economy)\b", "3E"),
    (r"\b(second sitting|2s|general)\b", "2S"),
]


@dataclass
class TravelRequest:
    raw: str
    src: str | None = None
    dst: str | None = None
    journey: date | None = None
    budget: float | None = None
    arrive_by: str | None = None
    depart_after: str | None = None
    classes: list[str] = field(default_factory=list)
    ac_only: bool = False
    passengers: int = 1
    objective: str = "balanced"     # cheapest | fastest | reliable | balanced
    allow_transfer: bool = True
    assumptions: list[str] = field(default_factory=list)

    def missing(self) -> list[str]:
        return [k for k in ("src", "dst") if getattr(self, k) is None]


def _parse_time(txt: str) -> str | None:
    m = re.match(r"(\d{1,2})(?:[:.](\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?", txt.strip(), re.I)
    if not m:
        return None
    h, mi, ap = int(m.group(1)), int(m.group(2) or 0), (m.group(3) or "").lower().replace(".", "")
    if ap == "pm" and h < 12:
        h += 12
    if ap == "am" and h == 12:
        h = 0
    if h > 23 or mi > 59:
        return None
    return f"{h:02d}:{mi:02d}"


def _find_cities(text: str) -> list[tuple[int, str]]:
    low = text.lower()
    found = []
    names = set(CITY_TO_STATIONS) | set(CITY_ALIASES)
    for n in sorted(names, key=len, reverse=True):
        for m in re.finditer(r"(?<![\w])" + re.escape(n) + r"(?![\w])", low):
            city = CITY_ALIASES.get(n, n)
            if not any(m.start() < e and s < m.end() for s, e, _ in found):
                found.append((m.start(), m.end(), city))
    for code in STATION_INDEX:
        for m in re.finditer(r"\b" + code + r"\b", text):
            found.append((m.start(), m.end(), STATION_INDEX[code][2].lower()))
    found.sort()
    out = []
    for s, _, c in found:
        if not out or out[-1][1] != c:
            out.append((s, c))
    return out


def parse_request(text: str, today: date | None = None) -> TravelRequest:
    """Rule-based NLU (English + Hinglish). Deterministic, auditable, no API key needed."""
    today = today or date.today()
    req = TravelRequest(raw=text)
    low = text.lower().replace("₹", " rs ").replace(",", "")
    # --- route
    m = re.search(r"from\s+([\w\sऀ-ॿঀ-৿]+?)\s+to\s+([\w\sऀ-ॿঀ-৿]+?)(?:\s+(?:on|by|for|under|below|tomorrow|today|in|with|before|after|within|next|reach|arriv|leav|depart|kal|parso)\b|[,.?!]|$)", low)
    cities = []
    if m:
        a, b = data.resolve_city(m.group(1)), data.resolve_city(m.group(2))
        if a and b:
            cities = [a, b]
    if not cities:
        cities = [c for _, c in _find_cities(text)]
    if len(cities) >= 2:
        req.src, req.dst = cities[0], cities[1]
    elif len(cities) == 1:
        req.dst = cities[0]
    # --- date
    if re.search(r"day after tomorrow|parso", low):
        req.journey = today + timedelta(days=2)
    elif re.search(r"tomorrow|\bkal\b|tmrw", low):
        req.journey = today + timedelta(days=1)
    elif "today" in low or "tonight" in low or "aaj" in low:
        req.journey = today
    else:
        m = re.search(r"in (\d{1,2}) days", low)
        if m:
            req.journey = today + timedelta(days=int(m.group(1)))
        m2 = re.search(r"(\d{1,2})(?:st|nd|rd|th)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*", low) or \
            re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{1,2})", low)
        m3 = re.search(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b", low)
        if m2:
            g = m2.groups()
            day, mon = (int(g[0]), MONTHS[g[1]]) if g[0].isdigit() else (int(g[1]), MONTHS[g[0]])
            y = today.year if (mon, day) >= (today.month, today.day) else today.year + 1
            req.journey = date(y, mon, day)
        elif m3 and not req.journey:
            day, mon = int(m3.group(1)), int(m3.group(2))
            y = int(m3.group(3)) if m3.group(3) else today.year
            y = y + 2000 if y < 100 else y
            try:
                req.journey = date(y, mon, day)
            except ValueError:
                pass
        for i, wd in enumerate(WEEKDAYS):
            if re.search(rf"\b{wd}\b", low):
                delta = (i - today.weekday()) % 7 or 7
                if re.search(rf"next\s+{wd}", low) and delta < 7:
                    delta += 7
                req.journey = today + timedelta(days=delta)
    if req.journey is None:
        req.journey = today + timedelta(days=1)
        req.assumptions.append("No date given — assumed tomorrow.")
    # --- budget
    m = re.search(r"(?:under|below|within|less than|max(?:imum)?|budget(?: of)?|upto|up to|<)\s*(?:rs\.?|inr|rupees)?\s*(\d{2,6})", low)
    if m:
        req.budget = float(m.group(1))
    # --- times
    m = re.search(r"(?:reach|arrive|arriving|be there|get there|pahunch\w*)[^0-9]{0,25}?(?:before|by|till|until)?\s*(\d{1,2}(?:[:.]\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)?)", low) or \
        re.search(r"before\s+(\d{1,2}(?:[:.]\d{2})?\s*(?:am|pm)?)", low)
    if m:
        req.arrive_by = _parse_time(m.group(1))
    m = re.search(r"(?:leave|depart|leaving|start)\w*\s+after\s+(\d{1,2}(?:[:.]\d{2})?\s*(?:am|pm)?)", low) or \
        re.search(r"after\s+(\d{1,2}(?:[:.]\d{2})?\s*(?:am|pm))", low)
    if m:
        req.depart_after = _parse_time(m.group(1))
    # --- class / pax / objective
    for pat, cls in CLASS_WORDS:
        if re.search(pat, low):
            req.classes.append(cls)
    if re.search(r"\bac\b", low) and not req.classes:
        req.ac_only = True
    m = re.search(r"(\d+)\s*(?:people|persons|passengers|pax|adults|travellers|travelers|log)", low)
    if m:
        req.passengers = max(1, min(int(m.group(1)), 6))
    if re.search(r"cheap|lowest fare|sasta|budget", low) and not req.budget:
        req.objective = "cheapest"
    elif re.search(r"fast|quick|earliest|shortest", low):
        req.objective = "fastest"
    elif re.search(r"reliable|on.?time|punctual", low):
        req.objective = "reliable"
    if re.search(r"direct|no change|without change", low):
        req.allow_transfer = False
    return req


class PlannerAgent:
    name = "travel_planner_agent"

    def __init__(self, today: date | None = None):
        self.today = today or date.today()
        self.trace: list[dict[str, Any]] = []
        self.run_id = uuid.uuid4().hex[:10]
        from .confirmation import ConfirmationPredictor
        from .delay import DelayPredictor
        from .fare import FarePredictor
        self.conf, self.delay, self.fare = ConfirmationPredictor(), DelayPredictor(), FarePredictor()
        self._delay_cache: dict = {}

    # ------------------------------------------------------------ tracing
    def _step(self, agent: str, tool: str, inp: Any, fn):
        t0 = time.perf_counter()
        status, out = "ok", None
        try:
            out = fn()
            return out
        except Exception as exc:  # noqa: BLE001
            status, out = "error", str(exc)
            raise
        finally:
            summary = out if isinstance(out, (str, int, float)) else (
                f"{len(out)} rows" if isinstance(out, (list, pd.DataFrame)) else type(out).__name__)
            self.trace.append({"step": len(self.trace) + 1, "agent": agent, "tool": tool, "input": str(inp)[:160],
                               "output": str(summary)[:160], "status": status,
                               "latency_ms": round((time.perf_counter() - t0) * 1000, 1)})

    # ----------------------------------------------------------- sub-agents
    def _delay(self, tn: str, d: date) -> float:
        key = (tn, d)
        if key not in self._delay_cache:
            rec = self.delay.build_record(tn, d)
            self._delay_cache[key] = max(float(self.delay.predict_frame(pd.DataFrame([rec]))[0]), 0.0)
        return self._delay_cache[key]

    def search(self, req: TravelRequest) -> pd.DataFrame:
        legs = []
        direct = data.trains_between(req.src, req.dst, req.journey)
        for _, t in direct.iterrows():
            legs.append({"type": "direct", "legs": [(t["train_no"], req.journey)]})
        if req.allow_transfer and len(direct) < 3:
            src_st, dst_st = data.stations_for(req.src), data.stations_for(req.dst)
            all_t = data.trains()
            first = all_t[all_t["src"].isin(src_st) & ~all_t["dst"].isin(dst_st)]
            for _, a in first.iterrows():
                if not data.runs_on(a["run_days"], req.journey):
                    continue
                arr_a = data.arrival_datetime(a, req.journey)
                hub_city = STATION_INDEX[a["dst"]][2]
                for _, b in data.trains_between(hub_city, req.dst).iterrows():
                    for d in (arr_a.date(), arr_a.date() + timedelta(days=1)):
                        if not data.runs_on(b["run_days"], d):
                            continue
                        dep_b = datetime.combine(d, datetime.min.time()) + timedelta(minutes=int(b["dep_min"]))
                        gap = (dep_b - arr_a).total_seconds() / 60
                        if 60 <= gap <= 14 * 60:
                            legs.append({"type": "1-change", "legs": [(a["train_no"], req.journey), (b["train_no"], d)],
                                         "layover_min": int(gap), "hub": a["dst"]})
        if faults.active("agent_loop"):
            for _ in range(GOVERNANCE["agent_max_tool_calls"]):
                self._step("SearchAgent", "search_trains(retry)", req.src, lambda: direct)
        return legs

    def _option_classes(self, train_no: str, req: TravelRequest) -> list[str]:
        cls = data.train(train_no)["classes"].split(",")
        if req.classes:
            cls = [c for c in cls if c in req.classes]
        elif req.ac_only:
            cls = [c for c in cls if CLASS_INFO[c][3]]
        return cls

    def evaluate(self, itins: list[dict], req: TravelRequest) -> pd.DataFrame:
        rows = []
        for it in itins:
            legs = it["legs"]
            cls_sets = [self._option_classes(tn, req) for tn, _ in legs]
            if not all(cls_sets):
                continue
            # evaluate each class of the (first) leg, cheapest compatible class on others
            for cls in cls_sets[0]:
                total_fare, conf_p, details, sched_arr, risk_arr, dep_dt = 0.0, 1.0, [], None, None, None
                ok = True
                for i, (tn, d) in enumerate(legs):
                    c = cls if i == 0 else (cls if cls in cls_sets[i] else
                                            min(cls_sets[i], key=lambda k: inventory.fare(tn, k)))
                    t = data.train(tn)
                    av = inventory.availability(tn, c, d, self.today)
                    p_conf = 1.0
                    if av["status"] != "AVAILABLE":
                        rec = self.conf.build_record(tn, c, d, av["count"], "RAC" if av["status"] == "RAC" else "WL",
                                                     booked_on=self.today)
                        p_conf = float(self.conf.predict_frame(pd.DataFrame([rec]))[0])
                    pred_delay = self._delay(tn, d)
                    f = inventory.fare(tn, c) * req.passengers
                    arr = data.arrival_datetime(t, d)
                    if i == 0:
                        dep_dt = datetime.combine(d, datetime.min.time()) + timedelta(minutes=int(t["dep_min"]))
                    total_fare += f
                    conf_p *= p_conf
                    sched_arr, risk_arr = arr, arr + timedelta(minutes=pred_delay)
                    details.append({"train_no": tn, "train_name": t["train_name"], "class": c, "date": d.isoformat(),
                                    "dep": t["dep_time"], "arr": t["arr_time"], "from": t["src"], "to": t["dst"],
                                    "fare": f, "availability": av["label"], "confirm_prob": round(p_conf, 3),
                                    "pred_delay_min": round(pred_delay, 0), "punctuality": t["punctuality"]})
                if not ok:
                    continue
                rows.append({"type": it["type"], "legs": details, "class": cls,
                             "trains": " → ".join(x["train_no"] for x in details),
                             "name": " + ".join(x["train_name"] for x in details),
                             "departure": dep_dt, "scheduled_arrival": sched_arr, "expected_arrival": risk_arr,
                             "duration_h": round((sched_arr - dep_dt).total_seconds() / 3600, 1),
                             "total_fare": total_fare, "confirm_prob": round(conf_p, 3),
                             "expected_delay_min": round((risk_arr - sched_arr).total_seconds() / 60, 0),
                             "layover_min": it.get("layover_min")})
        return pd.DataFrame(rows)

    def optimize(self, opts: pd.DataFrame, req: TravelRequest) -> tuple[pd.DataFrame, pd.DataFrame]:
        if opts.empty:
            return opts, opts
        o = opts.copy()
        o["violations"] = [[] for _ in range(len(o))]
        if req.budget:
            for i in o.index[o["total_fare"] > req.budget]:
                o.at[i, "violations"].append(f"fare ₹{o.at[i, 'total_fare']:.0f} > budget ₹{req.budget:.0f}")
        if req.arrive_by:
            h, m = map(int, req.arrive_by.split(":"))
            for i in o.index:
                arr = o.at[i, "scheduled_arrival"]
                dep = o.at[i, "departure"]
                deadline = datetime.combine(arr.date(), datetime.min.time()) + timedelta(hours=h, minutes=m)
                late_day = (arr.date() - dep.date()).days > 1
                if arr > deadline or late_day:
                    o.at[i, "violations"].append(f"scheduled arrival {arr:%H:%M} after {req.arrive_by}")
        if req.depart_after:
            h, m = map(int, req.depart_after.split(":"))
            for i in o.index:
                if o.at[i, "departure"].time() < datetime.min.replace(hour=h, minute=m).time():
                    o.at[i, "violations"].append(f"departs before {req.depart_after}")
        o["feasible"] = o["violations"].apply(len) == 0
        # on-time risk vs deadline using predicted delay
        if req.arrive_by:
            h, m = map(int, req.arrive_by.split(":"))
            o["deadline_risk"] = [
                "high" if ea > datetime.combine(sa.date(), datetime.min.time()) + timedelta(hours=h, minutes=m)
                else "low" for ea, sa in zip(o["expected_arrival"], o["scheduled_arrival"])]
        else:
            o["deadline_risk"] = "n/a"
        w = {"cheapest": (0.6, 0.15, 0.25, 0.0), "fastest": (0.15, 0.6, 0.25, 0.0),
             "reliable": (0.15, 0.15, 0.4, 0.3), "balanced": (0.35, 0.25, 0.3, 0.1)}[req.objective]
        fare_n = (o["total_fare"] - o["total_fare"].min()) / max(o["total_fare"].max() - o["total_fare"].min(), 1)
        dur_n = (o["duration_h"] - o["duration_h"].min()) / max(o["duration_h"].max() - o["duration_h"].min(), 0.1)
        o["score"] = (w[0] * (1 - fare_n) + w[1] * (1 - dur_n) + w[2] * o["confirm_prob"]
                      + w[3] * (1 - o["expected_delay_min"].clip(0, 180) / 180)
                      - 0.15 * (o["deadline_risk"] == "high") - 0.05 * (o["type"] != "direct"))
        o = o.sort_values(["feasible", "score"], ascending=[False, False]).reset_index(drop=True)
        return o[o["feasible"]].reset_index(drop=True), o[~o["feasible"]].reset_index(drop=True)

    def validate(self, plan: dict | None, req: TravelRequest) -> tuple[bool, list[str]]:
        """ValidatorAgent: independent re-check of the recommended plan against facts and constraints."""
        issues = []
        if plan is None:
            return True, []
        for leg in plan["legs"]:
            try:
                t = data.train(leg["train_no"])
            except KeyError:
                issues.append(f"train {leg['train_no']} does not exist")
                continue
            if not data.runs_on(t["run_days"], date.fromisoformat(leg["date"])):
                issues.append(f"train {leg['train_no']} does not run on {leg['date']}")
            if leg["class"] not in t["classes"].split(","):
                issues.append(f"class {leg['class']} not offered on {leg['train_no']}")
            if abs(inventory.fare(leg["train_no"], leg["class"]) * req.passengers - leg["fare"]) > 1:
                issues.append(f"fare mismatch on {leg['train_no']}")
        if req.budget and plan["total_fare"] > req.budget:
            issues.append("plan exceeds budget")
        src_ok = plan["legs"][0]["from"] in data.stations_for(req.src)
        dst_ok = plan["legs"][-1]["to"] in data.stations_for(req.dst)
        if not (src_ok and dst_ok):
            issues.append("plan does not connect requested cities")
        return not issues, issues

    # ------------------------------------------------------------------ run
    def plan(self, query: str) -> dict:
        t0 = time.perf_counter()
        req = self._step("IntentAgent", "parse_request", query, lambda: parse_request(query, self.today))
        if req.missing():
            return self._finish(query, t0, req, None, pd.DataFrame(), pd.DataFrame(), "clarify",
                                message=f"Please tell me the {' and '.join(req.missing())} city.")
        if req.src == req.dst:
            return self._finish(query, t0, req, None, pd.DataFrame(), pd.DataFrame(), "clarify",
                                message="Source and destination are the same city.")
        itins = self._step("SearchAgent", "search_itineraries", f"{req.src}->{req.dst} {req.journey}",
                           lambda: self.search(req))
        if not itins:
            alt_dates = []
            for k in (-1, 1, 2, 3):
                d = req.journey + timedelta(days=k)
                if d < self.today:
                    continue
                alt = TravelRequest(**{**req.__dict__, "journey": d, "assumptions": []})
                if self._step("SearchAgent", "search_itineraries", f"{req.src}->{req.dst} {d}", lambda: self.search(alt)):
                    alt_dates.append(f"{d:%a %d %b}")
            hint = f" Options exist on: {', '.join(alt_dates)}." if alt_dates else " Try another date."
            return self._finish(query, t0, req, None, pd.DataFrame(), pd.DataFrame(), "no_trains",
                                message=f"No trains found from {req.src.title()} to {req.dst.title()} on "
                                        f"{req.journey:%d %b %Y}.{hint}")
        opts = self._step("AvailabilityAgent+DelayAgent+FareAgent", "evaluate_options", f"{len(itins)} itineraries",
                          lambda: self.evaluate(itins, req))
        feasible, rejected = self._step("OptimizerAgent", "optimize", f"objective={req.objective}",
                                        lambda: self.optimize(opts, req))
        best = feasible.iloc[0].to_dict() if not feasible.empty else None
        valid, issues = self._step("ValidatorAgent", "validate_plan", best["trains"] if best else "-",
                                   lambda: self.validate(best, req))
        if best is not None and not valid:
            best = None
        if best is not None:
            msg = self._explain(best, req)
            fc = self._step("FareAgent", "fare_forecast", best["legs"][0]["train_no"],
                            lambda: self.fare.forecast(best["legs"][0]["train_no"], best["legs"][0]["class"],
                                                       date.fromisoformat(best["legs"][0]["date"]), self.today, log=False))
            if fc["trend"] == "rising":
                msg += f" Fares on this train are trending up (₹{fc['fare_now']:.0f} → ₹{fc['fare_at_journey']:.0f}), so booking now is advisable."
            outcome = "plan_ready"
        else:
            near = rejected.iloc[0] if not rejected.empty else None
            msg = "No option satisfies every constraint."
            if near is not None:
                msg += (f" Closest alternative: {near['name']} ({near['class']}) at ₹{near['total_fare']:.0f}, "
                        f"arriving {near['scheduled_arrival']:%d %b %H:%M} — blocked by: {'; '.join(near['violations'])}.")
            outcome = "infeasible"
        return self._finish(query, t0, req, best, feasible, rejected, outcome, message=msg, valid=valid,
                            validation_issues=issues)

    def _explain(self, b: dict, req: TravelRequest) -> str:
        legs = b["legs"]
        route = " then ".join(f"{l['train_name']} ({l['train_no']}, {l['class']}) dep {l['dep']} {l['date']}" for l in legs)
        s = (f"Best option: {route}. Total fare ₹{b['total_fare']:.0f} for {req.passengers} passenger(s); "
             f"scheduled arrival {b['scheduled_arrival']:%d %b %H:%M}, expected {b['expected_arrival']:%H:%M} "
             f"after a predicted {b['expected_delay_min']:.0f} min delay. Confirmation chance {b['confirm_prob']:.0%}.")
        if req.arrive_by and b["deadline_risk"] == "high":
            s += f" ⚠ Predicted delay may push arrival past {req.arrive_by}."
        if b["confirm_prob"] < 0.3:
            first = legs[0]
            if date.fromisoformat(first["date"]) - self.today == timedelta(days=1):
                opens = 10 if CLASS_INFO[first["class"]][3] else 11
                when = (f"opened today at {opens}:00 AM — check Tatkal quota now" if datetime.now().hour >= opens
                        else f"opens today at {opens}:00 AM — book then for a better chance")
                s += (f" ⚠ Current status is {first['availability']} with a low chance of confirmation; "
                      f"Tatkal for {first['class']} {when}.")
            else:
                s += (f" ⚠ Current status is {first['availability']} with a low chance of confirmation; "
                      "consider the next option or a different date.")
        return s

    def _finish(self, query, t0, req, best, feasible, rejected, outcome, message="", valid=True,
                validation_issues=None) -> dict:
        latency = (time.perf_counter() - t0) * 1000
        from ..governance.agent_monitor import detect_anomalies
        anomalies = detect_anomalies(self.trace, latency, outcome)
        store = Store.get()
        store.log_agent_run(self.name, self.run_id, query, len(self.trace), latency, outcome, valid, anomalies,
                            self.trace)
        store.audit(self.name, "agent.plan", outcome, {"run_id": self.run_id, "query": query, "steps": len(self.trace),
                                                       "best": best["trains"] if best else None,
                                                       "anomalies": anomalies}, "agent")
        return {"run_id": self.run_id, "request": req, "best": best, "options": feasible, "rejected": rejected,
                "outcome": outcome, "message": message, "trace": pd.DataFrame(self.trace), "latency_ms": latency,
                "valid": valid, "validation_issues": validation_issues or [], "anomalies": anomalies,
                "approval_token": uuid.uuid4().hex if best else None}


def book(plan_result: dict, approved_by: str, approval_token: str | None) -> list[dict]:
    """Executes booking for every leg — only with the token that was shown to the passenger."""
    if not approval_token or approval_token != plan_result.get("approval_token"):
        Store.get().audit("travel_planner_agent", "booking.denied", plan_result.get("run_id", ""),
                          {"reason": "invalid or missing approval token"}, "agent")
        raise PermissionError("Passenger approval required before booking")
    best = plan_result["best"]
    req = plan_result["request"]
    return [inventory.book(l["train_no"], l["class"], date.fromisoformat(l["date"]), req.passengers, approved_by,
                           approval_token) for l in best["legs"]]


def register_agent() -> str:
    cfg = {"type": "agentic-workflow", "sub_agents": ["IntentAgent", "SearchAgent", "AvailabilityAgent", "DelayAgent",
                                                      "FareAgent", "OptimizerAgent", "ValidatorAgent"],
           "max_tool_calls": GOVERNANCE["agent_max_tool_calls"], "booking_requires_approval": True}
    if registry.production_version(PlannerAgent.name):
        return registry.production_version(PlannerAgent.name)
    return registry.register(PlannerAgent.name, cfg, {"validity_rate": 1.0}, stage="production",
                             reason="initial agent configuration",
                             card={"description": "Agentic journey planner with validator and approval gate",
                                   "risk": "high", "owner": "AI Platform Team"})
