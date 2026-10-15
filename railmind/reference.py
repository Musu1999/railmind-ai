"""Static reference data: stations, festival calendar, class & fare rules.

All train data in this project is SYNTHETIC and generated for demonstration;
station codes and coordinates are real, train numbers/timings are not.
"""
from __future__ import annotations

import math
from datetime import date, timedelta

STATIONS = [
    # code, name, city, lat, lon, zone, base congestion (0-1)
    ("NDLS", "New Delhi", "Delhi", 28.6430, 77.2194, "NR", 0.95),
    ("NZM", "Hazrat Nizamuddin", "Delhi", 28.5880, 77.2510, "NR", 0.80),
    ("CSMT", "Mumbai CSMT", "Mumbai", 18.9400, 72.8355, "CR", 0.95),
    ("MMCT", "Mumbai Central", "Mumbai", 18.9696, 72.8194, "WR", 0.85),
    ("HWH", "Howrah Jn", "Kolkata", 22.5839, 88.3425, "ER", 0.95),
    ("SDAH", "Sealdah", "Kolkata", 22.5675, 88.3707, "ER", 0.85),
    ("MAS", "Chennai Central", "Chennai", 13.0827, 80.2750, "SR", 0.90),
    ("SBC", "KSR Bengaluru", "Bengaluru", 12.9784, 77.5713, "SWR", 0.88),
    ("SC", "Secunderabad Jn", "Hyderabad", 17.4337, 78.5016, "SCR", 0.85),
    ("PUNE", "Pune Jn", "Pune", 18.5286, 73.8743, "CR", 0.80),
    ("ADI", "Ahmedabad Jn", "Ahmedabad", 23.0258, 72.6010, "WR", 0.80),
    ("JP", "Jaipur Jn", "Jaipur", 26.9196, 75.7878, "NWR", 0.70),
    ("LKO", "Lucknow Charbagh", "Lucknow", 26.8317, 80.9180, "NR", 0.75),
    ("PNBE", "Patna Jn", "Patna", 25.6030, 85.1373, "ECR", 0.85),
    ("BPL", "Bhopal Jn", "Bhopal", 23.2672, 77.4126, "WCR", 0.70),
    ("NGP", "Nagpur Jn", "Nagpur", 21.1520, 79.0880, "CR", 0.70),
    ("BSB", "Varanasi Jn", "Varanasi", 25.3268, 82.9866, "NR", 0.75),
    ("GHY", "Guwahati", "Guwahati", 26.1820, 91.7510, "NFR", 0.65),
    ("BBS", "Bhubaneswar", "Bhubaneswar", 20.2665, 85.8432, "ECoR", 0.65),
    ("ERS", "Ernakulam Jn", "Kochi", 9.9690, 76.2900, "SR", 0.60),
    ("CDG", "Chandigarh", "Chandigarh", 30.7020, 76.8210, "NR", 0.55),
    ("MAO", "Madgaon", "Goa", 15.2667, 73.9667, "KR", 0.55),
    ("MAQ", "Mangaluru Central", "Mangaluru", 12.8630, 74.8370, "SR", 0.50),
    ("UD", "Udupi", "Udupi", 13.3410, 74.7520, "KR", 0.40),
]

STATION_INDEX = {s[0]: s for s in STATIONS}
CITY_TO_STATIONS: dict[str, list[str]] = {}
for _s in STATIONS:
    CITY_TO_STATIONS.setdefault(_s[2].lower(), []).append(_s[0])

# Common aliases people type in chat
CITY_ALIASES = {
    "new delhi": "delhi", "dilli": "delhi", "bombay": "mumbai", "calcutta": "kolkata",
    "madras": "chennai", "bangalore": "bengaluru", "bengalooru": "bengaluru",
    "secunderabad": "hyderabad", "cochin": "kochi", "ernakulam": "kochi",
    "banaras": "varanasi", "kashi": "varanasi", "mangalore": "mangaluru",
    "manipal": "udupi", "madgaon": "goa", "margao": "goa", "howrah": "kolkata",
    "sealdah": "kolkata", "दिल्ली": "delhi", "मुंबई": "mumbai", "पुणे": "pune",
    "कोलकाता": "kolkata", "पटना": "patna", "लखनऊ": "lucknow", "जयपुर": "jaipur",
    "কলকাতা": "kolkata", "দিল্লি": "delhi", "মুম্বাই": "mumbai",
}

# --------------------------------------------------------------- classes
CLASS_INFO = {
    # class: (label, fare per km INR, berths per coach, AC?)
    "2S": ("Second Sitting", 0.32, 108, False),
    "SL": ("Sleeper", 0.52, 72, False),
    "CC": ("AC Chair Car", 1.25, 78, True),
    "3E": ("AC 3 Economy", 1.30, 83, True),
    "3A": ("AC 3 Tier", 1.40, 64, True),
    "2A": ("AC 2 Tier", 2.05, 46, True),
    "EC": ("Executive Chair Car", 2.55, 56, True),
    "1A": ("AC First Class", 3.40, 24, True),
}
RESERVATION_CHARGE = {"2S": 15, "SL": 20, "CC": 40, "3E": 40, "3A": 40, "2A": 50, "EC": 60, "1A": 60}
SUPERFAST_CHARGE = {"2S": 15, "SL": 30, "CC": 45, "3E": 45, "3A": 45, "2A": 45, "EC": 75, "1A": 75}

TRAIN_TYPES = {
    # type: (avg speed km/h, classes, coaches per class, punctuality prior, flexi?)
    "Vande Bharat": (82, ["CC", "EC"], {"CC": 14, "EC": 2}, 0.90, False),
    "Rajdhani": (75, ["3A", "2A", "1A"], {"3A": 11, "2A": 5, "1A": 1}, 0.85, True),
    "Shatabdi": (72, ["CC", "EC"], {"CC": 12, "EC": 2}, 0.86, True),
    "Duronto": (70, ["SL", "3A", "2A", "1A"], {"SL": 6, "3A": 6, "2A": 3, "1A": 1}, 0.80, True),
    "Superfast": (60, ["SL", "3A", "2A"], {"SL": 10, "3A": 5, "2A": 2}, 0.72, False),
    "Express": (52, ["2S", "SL", "3E", "3A", "2A"], {"2S": 2, "SL": 10, "3E": 2, "3A": 4, "2A": 1}, 0.66, False),
    "Mail": (48, ["SL", "3A", "2A"], {"SL": 11, "3A": 4, "2A": 1}, 0.62, False),
}
SUPERFAST_TYPES = {"Vande Bharat", "Rajdhani", "Shatabdi", "Duronto", "Superfast"}
PREMIUM_MULTIPLIER = {"Vande Bharat": 1.25, "Rajdhani": 1.15, "Shatabdi": 1.10, "Duronto": 1.10}

# -------------------------------------------------------------- calendar
FESTIVALS = {
    "Holi": [date(2024, 3, 25), date(2025, 3, 14), date(2026, 3, 4), date(2027, 3, 22)],
    "Durga Puja": [date(2024, 10, 12), date(2025, 10, 2), date(2026, 10, 20)],
    "Diwali": [date(2024, 11, 1), date(2025, 10, 21), date(2026, 11, 8)],
    "Chhath": [date(2024, 11, 7), date(2025, 10, 28), date(2026, 11, 15)],
    "Christmas/New Year": [date(2024, 12, 28), date(2025, 12, 28), date(2026, 12, 28)],
}


def festival_intensity(d: date) -> float:
    """0..1 travel-demand intensity from festivals and holiday seasons."""
    best = 0.0
    for days in FESTIVALS.values():
        for f in days:
            gap = abs((d - f).days)
            if gap <= 10:
                best = max(best, 1.0 - gap / 11.0)
    if d.month in (5, 6):          # summer vacation
        best = max(best, 0.55)
    return round(best, 3)


def festival_name(d: date) -> str | None:
    for name, days in FESTIVALS.items():
        for f in days:
            if abs((d - f).days) <= 10:
                return name
    if d.month in (5, 6):
        return "Summer vacation"
    return None


def monsoon(d: date) -> bool:
    return d.month in (6, 7, 8, 9)


def fog_season(d: date) -> bool:
    return d.month in (12, 1, 2)


def haversine_km(a: str, b: str) -> float:
    s1, s2 = STATION_INDEX[a], STATION_INDEX[b]
    lat1, lon1, lat2, lon2 = map(math.radians, (s1[3], s1[4], s2[3], s2[4]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def rail_distance_km(a: str, b: str) -> int:
    """Rail routes are ~25 % longer than great-circle distance."""
    return int(round(haversine_km(a, b) * 1.25))


def base_fare(cls: str, distance_km: float, train_type: str) -> int:
    per_km = CLASS_INFO[cls][1]
    # telescopic taper: long distances are cheaper per km
    eff_km = distance_km if distance_km <= 500 else 500 + (distance_km - 500) * 0.82
    fare = per_km * eff_km + RESERVATION_CHARGE[cls]
    if train_type in SUPERFAST_TYPES:
        fare += SUPERFAST_CHARGE[cls]
    fare *= PREMIUM_MULTIPLIER.get(train_type, 1.0)
    return int(round(fare / 5.0) * 5)


def daterange(start: date, days: int):
    for i in range(days):
        yield start + timedelta(days=i)
