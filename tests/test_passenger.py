from datetime import date, timedelta

import pytest

from railmind import data
from railmind.passenger.base import GovernanceBlocked
from railmind.passenger.confirmation import ConfirmationPredictor
from railmind.passenger.delay import DelayPredictor
from railmind.passenger.fare import FarePredictor
from railmind.passenger.fraud import FraudDetector
from railmind.passenger.planner import PlannerAgent, book, parse_request
from railmind.passenger.recommender import TrainRecommender

TOMORROW = date.today() + timedelta(days=1)


def test_reference_data_has_flagship_corridor():
    assert len(data.trains_between("Pune", "Delhi")) >= 3
    assert data.resolve_city("Bombay") == "mumbai"
    assert data.resolve_city("NDLS") == "delhi"


def test_confirmation_prediction_is_explained_and_bounded():
    r = ConfirmationPredictor().predict("12089", "SL", date.today() + timedelta(days=20), 15)
    assert 0 <= r["probability"] <= 1
    assert set(r["status_distribution"]) <= {"CNF", "RAC", "WL"}
    assert len(r["explanation"]) == len(ConfirmationPredictor().features)


def test_data_quality_gate_blocks_impossible_input():
    svc = ConfirmationPredictor()
    rec = svc.build_record("12089", "SL", TOMORROW, 10)
    rec["festival_intensity"] = 7.0          # outside physically valid [0, 1]
    with pytest.raises(GovernanceBlocked):
        svc.infer(rec)


def test_delay_interval_contains_point_forecast():
    r = DelayPredictor().predict("12089", TOMORROW, weather="fog")
    lo, hi = r["interval_80"]
    assert lo <= r["delay_min"] <= hi
    assert 0 <= r["on_time_probability"] <= 1


def test_fare_forecast_and_booking_window():
    r = FarePredictor().forecast("12001", "3A", date.today() + timedelta(days=30))
    assert len(r["curve"]) == 31
    assert r["best_fare"] <= r["curve"]["fare"].max()


def test_fraud_detector_flags_bot():
    bot = dict(account_age_days=5, bookings_last_24h=30, passengers_per_booking=6, session_duration_s=30,
               avg_click_interval_ms=120, form_fill_time_s=1.5, captcha_failures=4, payment_failures=0,
               distinct_cards_7d=1, accounts_per_device=1, ip_reputation=0.1, is_tatkal=1, booking_hour=10,
               geo_mismatch=0, amount=7000)
    r = FraudDetector().risk(bot)
    assert r["decision"] in ("BLOCK", "REVIEW")
    assert "bot_click_speed" in r["rule_ids"]


def test_recommender_ranks_with_reasons():
    r = TrainRecommender().recommend("Mumbai", "Delhi", date.today() + timedelta(days=14))
    assert len(r["items"]) > 0
    assert r["items"]["score"].is_monotonic_decreasing
    assert all(r["items"]["why"].str.len() > 0)


def test_planner_understands_flagship_query():
    q = "I want to travel from Pune to Delhi tomorrow under ₹1500 and reach before 10:00 AM."
    req = parse_request(q)
    assert (req.src, req.dst, req.journey, req.budget, req.arrive_by) == ("pune", "delhi", TOMORROW, 1500.0, "10:00")
    r = PlannerAgent().plan(q)
    assert r["outcome"] == "plan_ready" and r["valid"]
    assert r["best"]["total_fare"] <= 1500
    assert r["best"]["scheduled_arrival"].hour < 10


def test_planner_never_books_without_approval():
    r = PlannerAgent().plan("Pune to Delhi tomorrow under 1500")
    with pytest.raises(PermissionError):
        book(r, approved_by="test", approval_token=None)
    assert book(r, approved_by="test", approval_token=r["approval_token"])[0]["pnr"]


def test_hinglish_parsing():
    req = parse_request("kal Kolkata se Patna jana hai sasta")
    assert req.src == "kolkata" and req.dst == "patna" and req.journey == TOMORROW and req.objective == "cheapest"
