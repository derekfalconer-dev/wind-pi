
from flask import Flask, jsonify, request, Response
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import io
import math
import threading
import time
from calypso_reader import CalypsoReader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dateutil import parser


app = Flask(__name__)

reader = CalypsoReader()

LATEST_WIND = {
    "timestamp": None,
    "wind_speed_knots": 0.0,
    "wind_direction_degrees": 0.0,
    "wind_direction_cardinal": "--",
    "raw_sentence": "",
}

ACTIVE_BUCKET = {
    "bucket_start": None,
    "samples": [],  # list of {"ts": datetime, "speed": float, "direction": float}
}

DB_PATH = Path("/home/pi/wind_server/wind.db")
PACIFIC = ZoneInfo("America/Los_Angeles")
STALE_SECONDS = 60
DIRECTION_OFFSET_DEGREES = 287
HISTORY_DAYS = 30
PERCENTILE_MIN_DAYS = 7

def apply_direction_offset(raw_direction: float) -> float:
    return (raw_direction + DIRECTION_OFFSET_DEGREES) % 360


_plot_cache = {
    "ts": None,
    "json": None,
    "png": None,
}


def now_pacific():
    return datetime.now(PACIFIC)


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS wind_buckets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bucket_start TEXT NOT NULL,
                bucket_end TEXT NOT NULL,
                avg_wind_knots REAL NOT NULL,
                gust_knots REAL NOT NULL,
                avg_direction_degrees REAL,
                sample_count INTEGER NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_wind_buckets_start
            ON wind_buckets(bucket_start)
        """)
        conn.commit()


def db_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def insert_bucket(bucket_start: datetime,
                  bucket_end: datetime,
                  avg_wind_knots: float,
                  gust_knots: float,
                  avg_direction_degrees: float | None,
                  sample_count: int):
    with db_conn() as conn:
        conn.execute(
            """
            INSERT INTO wind_buckets (
                bucket_start,
                bucket_end,
                avg_wind_knots,
                gust_knots,
                avg_direction_degrees,
                sample_count
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                bucket_start.isoformat(),
                bucket_end.isoformat(),
                avg_wind_knots,
                gust_knots,
                avg_direction_degrees,
                sample_count,
            ),
        )
        conn.commit()


def latest_bucket():
    with db_conn() as conn:
        row = conn.execute("""
            SELECT bucket_start, bucket_end, avg_wind_knots, gust_knots,
                   avg_direction_degrees, sample_count
            FROM wind_buckets
            ORDER BY bucket_start DESC
            LIMIT 1
        """).fetchone()
    return row


def buckets_since(dt: datetime):
    with db_conn() as conn:
        rows = conn.execute("""
            SELECT bucket_start, bucket_end, avg_wind_knots, gust_knots,
                   avg_direction_degrees, sample_count
            FROM wind_buckets
            WHERE bucket_start >= ?
            ORDER BY bucket_start ASC
        """, (dt.isoformat(),)).fetchall()
    return rows


def buckets_between(start_dt: datetime, end_dt: datetime):
    with db_conn() as conn:
        rows = conn.execute("""
            SELECT bucket_start, bucket_end, avg_wind_knots, gust_knots,
                   avg_direction_degrees, sample_count
            FROM wind_buckets
            WHERE bucket_start >= ? AND bucket_start < ?
            ORDER BY bucket_start ASC
        """, (start_dt.isoformat(), end_dt.isoformat())).fetchall()
    return rows



def cardinal_from_degrees(deg: float) -> str:
    directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                  "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = round((deg % 360) / 22.5) % 16
    return directions[idx]


def circular_mean_degrees(degrees_list):
    if not degrees_list:
        return None
    sin_sum = sum(math.sin(math.radians(d)) for d in degrees_list)
    cos_sum = sum(math.cos(math.radians(d)) for d in degrees_list)
    if abs(sin_sum) < 1e-9 and abs(cos_sum) < 1e-9:
        return None
    angle = math.degrees(math.atan2(sin_sum, cos_sum)) % 360
    return angle


def serial_loop():
    global LATEST_WIND

    global ACTIVE_BUCKET

    active_bucket_start = None
    active_speeds = []
    active_directions = []
    active_samples = []
    last_cleanup = 0.0

    while True:
        try:
            reading = reader.read_once()
            if reading is None:
                continue

            reading_ts = now_pacific()
            direction = reading["wind_direction_deg"]
            speed_knots = reading["wind_speed_kt"]
            bucket_start = floor_to_5min(reading_ts)

            LATEST_WIND = {
                "timestamp": reading["timestamp"],
                "wind_speed_knots": speed_knots,
                "wind_direction_degrees": direction,
                "wind_direction_cardinal": cardinal_from_degrees(direction),
                "raw_sentence": reading["raw_sentence"],
            }

            if active_bucket_start is None:
                active_bucket_start = bucket_start

            if bucket_start != active_bucket_start:
                avg_wind = trimmed_mean(active_speeds)
                gust_wind = robust_gust(active_speeds)
                avg_direction = circular_mean_degrees(active_directions)

                if avg_wind is not None and gust_wind is not None:
                    insert_bucket(
                        bucket_start=active_bucket_start,
                        bucket_end=active_bucket_start + timedelta(minutes=5),
                        avg_wind_knots=round(avg_wind, 3),
                        gust_knots=round(gust_wind, 3),
                        avg_direction_degrees=round(avg_direction, 3) if avg_direction is not None else None,
                        sample_count=len(active_speeds),
                    )

                active_bucket_start = bucket_start
                active_speeds = []
                active_directions = []
                active_samples = []            

            active_speeds.append(speed_knots)
            active_directions.append(direction)
            active_samples.append({
                "ts": reading_ts,
                "speed": speed_knots,
                "direction": direction,
            })

            ACTIVE_BUCKET = {
                "bucket_start": active_bucket_start,
                "samples": active_samples.copy(),
            }

            now_ts = time.time()
            if now_ts - last_cleanup >= 3600:
                cleanup_old_data()
                last_cleanup = now_ts

        except Exception as e:
            print(f"Serial read error: {e}")
            time.sleep(1)


def floor_to_5min(dt: datetime) -> datetime:
    return dt.replace(minute=(dt.minute // 5) * 5, second=0, microsecond=0)


def cleanup_old_data():
    cutoff = now_pacific() - timedelta(days=HISTORY_DAYS)
    with db_conn() as conn:
        conn.execute("DELETE FROM wind_buckets WHERE bucket_start < ?", (cutoff.isoformat(),))
        conn.commit()


def summarize_window(minutes: int):
    cutoff = now_pacific() - timedelta(minutes=minutes)
    rows = buckets_since(cutoff)

    if not rows:
        return {
            "count": 0,
            "avg_speed": None,
            "gust": None,
            "avg_direction": None,
        }

    avg_speeds = [row["avg_wind_knots"] for row in rows]
    gusts = [row["gust_knots"] for row in rows]
    directions = [
        row["avg_direction_degrees"]
        for row in rows
        if row["avg_direction_degrees"] is not None
    ]

    return {
        "count": len(rows),
        "avg_speed": round(sum(avg_speeds) / len(avg_speeds), 1),
        "gust": round(max(gusts), 1),
        "avg_direction": circular_mean_degrees(directions) if directions else None,
    }


def floor_to_5min(dt: datetime) -> datetime:
    return dt.replace(minute=(dt.minute // 5) * 5, second=0, microsecond=0)


def trimmed_mean(values):
    if not values:
        return None

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    trim = n // 4
    middle_vals = sorted_vals[trim:n - trim]

    if not middle_vals:
        middle_vals = sorted_vals

    return sum(middle_vals) / len(middle_vals)


def robust_gust(values):
    if not values:
        return None

    sorted_vals = sorted(values)
    n = len(sorted_vals)

    top_count = max(1, int(round(n * 0.10)))
    top_vals = sorted_vals[-top_count:]

    if len(top_vals) >= 4:
        trim_top = len(top_vals) // 4
        trimmed_top_vals = top_vals[trim_top:len(top_vals) - trim_top]
        if trimmed_top_vals:
            return sum(trimmed_top_vals) / len(trimmed_top_vals)

    return sum(top_vals) / len(top_vals)


def summarize_live_window(seconds: int):
    samples = ACTIVE_BUCKET.get("samples", [])
    if not samples:
        return {
            "count": 0,
            "avg_speed": None,
            "gust": None,
            "avg_direction": None,
        }

    cutoff = now_pacific() - timedelta(seconds=seconds)
    live_samples = [s for s in samples if s["ts"] >= cutoff]

    if not live_samples:
        return {
            "count": 0,
            "avg_speed": None,
            "gust": None,
            "avg_direction": None,
        }

    speeds = [s["speed"] for s in live_samples]
    directions = [s["direction"] for s in live_samples]

    return {
        "count": len(live_samples),
        "avg_speed": round(trimmed_mean(speeds), 1) if speeds else None,
        "gust": round(robust_gust(speeds), 1) if speeds else None,
        "avg_direction": circular_mean_degrees(directions) if directions else None,
    }


def summarize_live_rolling_5min():
    samples = ACTIVE_BUCKET.get("samples", [])
    if not samples:
        latest = latest_bucket()
        if not latest:
            return {
                "count": 0,
                "avg_speed": None,
                "gust": None,
                "avg_direction": None,
            }
        return {
            "count": 1,
            "avg_speed": round(latest["avg_wind_knots"], 1),
            "gust": round(latest["gust_knots"], 1),
            "avg_direction": latest["avg_direction_degrees"],
        }

    speeds = [s["speed"] for s in samples]
    directions = [s["direction"] for s in samples]

    return {
        "count": len(samples),
        "avg_speed": round(trimmed_mean(speeds), 1) if speeds else None,
        "gust": round(robust_gust(speeds), 1) if speeds else None,
        "avg_direction": circular_mean_degrees(directions) if directions else None,
    }


def bucket_metric_and_direction(rows):
    if not rows:
        return None, None

    avg_vals = [float(r["avg_wind_knots"]) for r in rows]
    dir_vals = [
        float(r["avg_direction_degrees"])
        for r in rows
        if r["avg_direction_degrees"] is not None
    ]

    speed_metric = sum(avg_vals) / len(avg_vals) if avg_vals else None
    avg_direction = circular_mean_degrees(dir_vals) if dir_vals else None

    return speed_metric, avg_direction


def percentile_rank(value, history_values):
    if value is None or not history_values:
        return None

    sorted_values = sorted(float(x) for x in history_values)
    n = len(sorted_values)

    below = sum(1 for x in sorted_values if x < value)
    equal = sum(1 for x in sorted_values if x == value)

    # Midrank percentile:
    # - lowest of 8 days displays around 6th percentile, not 0th
    # - highest of 8 days displays around 94th percentile, not 100th
    pct = 100.0 * (below + 0.5 * equal) / n

    return round(pct, 0)


def direction_regime_from_cardinal(cardinal: str) -> str:
    if cardinal in {"N", "NNW", "NW", "WNW", "W"}:
        return cardinal
    return "OTHER"


def last_completed_bucket_window():
    bucket_end = floor_to_5min(now_pacific())
    bucket_start = bucket_end - timedelta(minutes=5)
    return bucket_start, bucket_end


def current_bucket_percentile():
    bucket_start, bucket_end = last_completed_bucket_window()
    current_rows = buckets_between(bucket_start, bucket_end)
    current_value, current_direction = bucket_metric_and_direction(current_rows)

    if current_value is None:
        return {
            "bucket_start": bucket_start,
            "bucket_end": bucket_end,
            "current_value": None,
            "percentile": None,
            "sample_days": 0,
            "direction_cardinal": None,
            "direction_regime": "OTHER",
        }

    time_of_day_key = bucket_start.strftime("%H:%M")
    history_values = []
    history_days = set()

    for day_offset in range(1, HISTORY_DAYS + 1):
        hist_start = bucket_start - timedelta(days=day_offset)
        hist_end = bucket_end - timedelta(days=day_offset)

        hist_rows = buckets_between(hist_start, hist_end)
        hist_value, _ = bucket_metric_and_direction(hist_rows)

        if hist_value is not None:
            history_values.append(hist_value)
            history_days.add(hist_start.date().isoformat())

    direction_cardinal = cardinal_from_degrees(current_direction) if current_direction is not None else None
    direction_regime = direction_regime_from_cardinal(direction_cardinal) if direction_cardinal else "OTHER"

    percentile = percentile_rank(current_value, history_values) if len(history_values) >= PERCENTILE_MIN_DAYS else None

    return {
        "bucket_start": bucket_start,
        "bucket_end": bucket_end,
        "time_of_day_key": time_of_day_key,
        "current_value": round(current_value, 1) if current_value is not None else None,
        "percentile": percentile,
        "sample_days": len(history_days),
        "direction_cardinal": direction_cardinal,
        "direction_regime": direction_regime,
    }


def foiling_window_auc_percentile():
    """
    Compare today's wind exposure from 09:00 to now against prior days.

    Active window:
      09:00 <= now < 21:00

    Metric:
      Sum of 5-minute avg wind buckets from 09:00 to current completed bucket.
      Since all buckets are 5 minutes, this is proportional to area under the curve.
    """
    local_now = now_pacific()
    today_9am = local_now.replace(hour=9, minute=0, second=0, microsecond=0)
    today_9pm = local_now.replace(hour=21, minute=0, second=0, microsecond=0)

    if local_now < today_9am or local_now >= today_9pm:
        return {
            "status": "night",
            "label": "NIGHT TIME",
            "current_value": None,
            "percentile": None,
            "sample_days": 0,
            "window_start": today_9am,
            "window_end": today_9pm,
        }

    _, current_bucket_cutoff = last_completed_bucket_window()

    # Do not let the comparison window run past 21:00.
    current_bucket_cutoff = min(current_bucket_cutoff, today_9pm)

    today_rows = buckets_between(today_9am, current_bucket_cutoff)
    today_values = [
        float(row["avg_wind_knots"])
        for row in today_rows
        if row["avg_wind_knots"] is not None
    ]

    if not today_values:
        return {
            "status": "building",
            "label": "BUILDING",
            "current_value": None,
            "percentile": None,
            "sample_days": 0,
            "window_start": today_9am,
            "window_end": current_bucket_cutoff,
        }

    # AUC in knot-hours.
    today_auc = sum(today_values) * (5.0 / 60.0)

    # Average wind across completed 5-minute buckets.
    today_avg_wind = sum(today_values) / len(today_values)

    history_values = []
    history_days = set()

    expected_bucket_count = len(today_values)
    min_bucket_count = max(1, int(expected_bucket_count * 0.80))

    for day_offset in range(1, HISTORY_DAYS + 1):
        hist_start = today_9am - timedelta(days=day_offset)
        hist_cutoff = current_bucket_cutoff - timedelta(days=day_offset)

        hist_rows = buckets_between(hist_start, hist_cutoff)
        hist_values = [
            float(row["avg_wind_knots"])
            for row in hist_rows
            if row["avg_wind_knots"] is not None
        ]

        # Fair comparison: only compare against historical days that have
        # roughly the same 9am-to-current-time coverage.
        if len(hist_values) >= min_bucket_count:
            history_values.append(sum(hist_values) * (5.0 / 60.0))
            history_days.add(hist_start.date().isoformat())

    percentile = (
        percentile_rank(today_auc, history_values)
        if len(history_values) >= PERCENTILE_MIN_DAYS
        else None
    )

    elapsed_hours = (current_bucket_cutoff - today_9am).total_seconds() / 3600.0
    avg_window_wind = today_avg_wind if today_values else None

    return {
        "status": "active",
        "label": "ACTIVE",
        "current_value": round(avg_window_wind, 1) if avg_window_wind is not None else None,
        "auc_value": round(today_auc, 1),
        "elapsed_hours": round(elapsed_hours, 1),
        "percentile": percentile,
        "sample_days": len(history_days),
        "window_start": today_9am,
        "window_end": current_bucket_cutoff,
    }


def today_bucket_debug_rows():
    local_now = now_pacific()
    start = local_now.replace(hour=9, minute=0, second=0, microsecond=0)

    _, cutoff = last_completed_bucket_window()
    rows = buckets_between(start, cutoff)

    result = []

    for r in rows:
        ts = datetime.fromisoformat(
            r["bucket_start"]
        ).astimezone(PACIFIC)

        result.append({
            "time": ts.strftime("%H:%M"),
            "avg": round(float(r["avg_wind_knots"]),2),
            "gust": round(float(r["gust_knots"]),2),
        })

    return result


def foiling_day_signal(bucket_pct, day_pct, direction_regime):
    if day_pct is None:
        return "BUILDING HISTORY"

    if day_pct >= 80 and direction_regime in {"N", "NNW", "NW", "WNW", "W"}:
        return "GO"
    if day_pct >= 65 and direction_regime in {"N", "NNW", "NW", "WNW", "W"}:
        return "BORDERLINE"
    if day_pct >= 80 and direction_regime == "OTHER":
        return "CAUTION"
    if bucket_pct is not None and bucket_pct >= 80 and direction_regime in {"N", "NNW", "NW", "WNW", "W"}:
        return "WATCH"
    return "NO GO"


def max_gust_today():
    local_now = now_pacific()
    start_of_day = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
    rows = buckets_since(start_of_day)
    if not rows:
        return None
    return round(max(row["gust_knots"] for row in rows), 1)


def build_latest_payload():
    latest_ts_value = LATEST_WIND.get("timestamp")
    if latest_ts_value is None:
        return {
            "status": "no_data",
            "message": "No wind readings received yet."
        }

    if isinstance(latest_ts_value, (int, float)):
        ts = datetime.fromtimestamp(latest_ts_value, tz=PACIFIC)
    elif isinstance(latest_ts_value, str):
        ts = parser.isoparse(latest_ts_value).astimezone(PACIFIC)
    else:
        return {
            "status": "no_data",
            "message": f"Unexpected timestamp type: {type(latest_ts_value).__name__}"
        }    

    age_sec = (now_pacific() - ts).total_seconds()
    live_status = "LIVE" if age_sec <= STALE_SECONDS else "STALE"

    one_min = summarize_live_window(60)
    five_min = summarize_live_rolling_5min()    

    direction = float(LATEST_WIND["wind_direction_degrees"])
    current_cardinal = cardinal_from_degrees(direction)

    avg_dir_5m = five_min["avg_direction"]
    avg_dir_5m_cardinal = cardinal_from_degrees(avg_dir_5m) if avg_dir_5m is not None else None

    return {
        "status": "ok",
        "live_status": live_status,
        "last_update": ts.isoformat(timespec="seconds"),
        "wind_speed_knots": round(float(LATEST_WIND["wind_speed_knots"]), 1),
        "wind_direction_degrees": round(direction, 1),
        "wind_direction_cardinal": current_cardinal,
        "one_min_avg_knots": one_min["avg_speed"],
        "one_min_gust_knots": one_min["gust"],
        "five_min_avg_knots": five_min["avg_speed"],
        "five_min_avg_direction_degrees": round(avg_dir_5m, 1) if avg_dir_5m is not None else None,
        "five_min_avg_direction_cardinal": avg_dir_5m_cardinal,
        "max_gust_today_knots": max_gust_today(),
    }


@app.route("/")
@app.route("/dashboard")
def dashboard():
    payload = build_latest_payload()

    if payload["status"] != "ok":
        html = """
        <html>
        <head>
          <meta http-equiv="refresh" content="10">
          <title>Wind Pi Dashboard</title>
          <style>
            body { font-family: Arial, sans-serif; margin: 30px; }
          </style>
        </head>
        <body>
          <h1>Wind Pi Dashboard</h1>
          <p>No data yet.</p>
        </body>
        </html>
        """
        return Response(html, mimetype="text/html")

    bucket_pct = current_bucket_percentile()

    day_pct = foiling_window_auc_percentile()
    debug_rows = today_bucket_debug_rows()

    foil_signal = foiling_day_signal(
        bucket_pct["percentile"],
        day_pct["percentile"],
        bucket_pct["direction_regime"],
    )

    last_update_dt = parser.isoparse(payload["last_update"])
    now_dt = datetime.now(last_update_dt.tzinfo)

    seconds_ago = int((now_dt - last_update_dt).total_seconds())

    if seconds_ago < 60:
        last_update_human = f"{seconds_ago} sec ago"
    elif seconds_ago < 3600:
        last_update_human = f"{seconds_ago // 60} min ago"
    else:
        last_update_human = f"{seconds_ago // 3600} hr ago"

    payload["last_update_human"] = last_update_human

    status_color = "#198754" if payload["live_status"] == "LIVE" else "#dc3545"

    html = f"""
    <html>
    <head>
      <meta http-equiv="refresh" content="15">
      <title>Wind Pi Dashboard</title>
      <style>
        body {{
          font-family: Arial, sans-serif;
          margin: 24px;
          max-width: 1100px;
        }}
        .row {{
          display: flex;
          gap: 16px;
          flex-wrap: wrap;
          margin-bottom: 20px;
        }}
        .card {{
          border: 1px solid #ddd;
          border-radius: 10px;
          padding: 16px;
          min-width: 180px;
          box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }}
        .big {{
          font-size: 2rem;
          font-weight: bold;
        }}
        .label {{
          color: #666;
          margin-bottom: 8px;
        }}
        .subtext {{
          color: #666;
          font-size: 0.9rem;
          margin-top: 8px;
          line-height: 1.35;
        }}
        .status {{
          font-weight: bold;
          color: {status_color};
        }}
        table {{
          border-collapse: collapse;
          width: 100%;
          max-width: 700px;
        }}
        th, td {{
          border: 1px solid #ddd;
          padding: 10px;
          text-align: left;
        }}
        th {{
          background: #f6f6f6;
        }}
        .chart-wrap {{
          width: 100%;
          max-width: 1000px;
          height: 420px;
          border: 1px solid #ddd;
          border-radius: 10px;
          padding: 12px;
          box-sizing: border-box;
        }}
        #windChart {{
          width: 100%;
          height: 100%;
        }}
       .compass {{
         position: relative;
         margin-top: 10px;
         height: 70px;
         width: 70px;
         border: 1px solid #ddd;
         border-radius: 50%;
        }}
       .arrow {{
         position: absolute;
         top: 50%;
         left: 50%;
         transform: translate(-50%, -50%);
         transform-origin: center;
         font-size: 2rem;
         transition: transform 0.5s ease;
        }}
       .north {{
         position: absolute;
         top: 4px;
         left: 50%;
         transform: translateX(-50%);
         font-size: 0.8rem;
         color: #888;
        }}
      </style>
    </head>
    <body>
      <h1>Wind Pi Dashboard</h1>
      <p><span class="status">{payload["live_status"]}</span> · Last update: {payload["last_update"]} ({payload["last_update_human"]})</p>

      <div class="row">
        <div class="card">
          <div class="label">Current Wind</div>
          <div class="big">{payload["wind_speed_knots"]} kt</div>
        </div>

        <div class="card">
          <div class="label">Current Direction</div>
          <div class="big">
            {payload['wind_direction_degrees']}° {payload['wind_direction_cardinal']}
          </div>
          <div class="compass">
            <div class="arrow" style="transform: translate(-50%, -50%) rotate({payload['wind_direction_degrees'] - 90}deg);">➤</div>
            <div class="north">N</div>
          </div>
        </div>

        <div class="card">
          <div class="label">1-Min Avg</div>
          <div class="big">{payload["one_min_avg_knots"] if payload["one_min_avg_knots"] is not None else "--"} kt</div>
        </div>

        <div class="card">
          <div class="label">1-Min Gust</div>
          <div class="big">{payload["one_min_gust_knots"] if payload["one_min_gust_knots"] is not None else "--"} kt</div>
        </div>
        <div class="card">
          <div class="label">5-Min Avg (Rolling)</div>
          <div class="big">{payload["five_min_avg_knots"] if payload["five_min_avg_knots"] is not None else "--"} kt</div>
        </div>

        <div class="card">
          <div class="label">Foiling Day Signal</div>
          <div class="big">{foil_signal}</div>
          <div class="subtext">
            Foiling window: {day_pct["label"] if day_pct["status"] == "night" else (str(int(day_pct["percentile"])) + "th pct" if day_pct["percentile"] is not None else "building history")}<br>
            Current bucket: {str(int(bucket_pct["percentile"])) + "th pct" if bucket_pct["percentile"] is not None else "building history"}<br>
            Regime: {bucket_pct["direction_regime"]}
          </div>
        </div>

        <div class="card">
          <div class="label">Max Gust Today</div>
          <div class="big">{payload["max_gust_today_knots"] if payload["max_gust_today_knots"] is not None else "--"} kt</div>
        </div>
       </div>
      <h2>Summary</h2>
      <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Current Direction</td><td>{payload["wind_direction_degrees"]}° {payload["wind_direction_cardinal"]}</td></tr>
        <tr><td>5-Min Avg Direction</td><td>{payload["five_min_avg_direction_degrees"] if payload["five_min_avg_direction_degrees"] is not None else "--"} {("° " + payload["five_min_avg_direction_cardinal"]) if payload["five_min_avg_direction_cardinal"] else ""}</td></tr>
        <tr><td>Foiling Day Signal</td><td>{foil_signal}</td></tr>
        <tr><td>Direction Regime</td><td>{bucket_pct["direction_regime"]}</td></tr>
        <tr><td>Current Bucket Percentile</td><td>{str(int(bucket_pct["percentile"])) + "th" if bucket_pct["percentile"] is not None else "Building history"}</td></tr>
        <tr><td>Current Bucket Metric</td><td>{str(bucket_pct["current_value"]) + " kt" if bucket_pct["current_value"] is not None else "--"}</td></tr>
        <tr><td>Current Bucket History Days</td><td>{bucket_pct["sample_days"]}</td></tr>
        <tr><td>Foiling Window Day Percentile</td><td>{day_pct["label"] if day_pct["status"] == "night" else (str(int(day_pct["percentile"])) + "th" if day_pct["percentile"] is not None else "Building history")}</td></tr>
        <tr><td>Foiling Window Day Avg Wind</td><td>{str(day_pct["current_value"]) + " kt" if day_pct["current_value"] is not None else "--"}</td></tr>
        <tr><td>Foiling Window Day AUC (kn-hrs) </td><td>{str(day_pct.get("auc_value")) if day_pct.get("auc_value") is not None else "--"}</td></tr>
        <tr><td>Foiling Window History Days</td><td>{day_pct["sample_days"]}</td></tr>
        <tr><td>Last Update</td><td>{payload["last_update"]}</td></tr>
        <tr><td>Status</td><td>{payload["live_status"]}</td></tr>
      </table>
      <h2>Last 24 Hours</h2>
      <p>Trimmed-mean wind and gust wind by 5-minute interval.</p>
      <div class="chart-wrap">
        <canvas id="windChart"></canvas>
      </div>

      <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
      <script>
        async function loadWindChart() {{
          const resp = await fetch('/plot.json', {{ cache: 'no-store' }});
          const plotData = await resp.json();

          const ctx = document.getElementById('windChart').getContext('2d');

          new Chart(ctx, {{
            type: 'line',
            data: {{
              labels: plotData.labels,
              datasets: [
                {{
                  label: 'Trimmed Mean Wind',
                  data: plotData.trimmed_mean_knots,
                  borderWidth: 2,
                  pointRadius: 0,
                  tension: 0.15
                }},
                {{
                  label: 'Gust Wind',
                  data: plotData.gust_knots,
                  borderWidth: 1.5,
                  pointRadius: 0,
                  tension: 0.10
                }}
              ]
            }},
            options: {{
              responsive: true,
              maintainAspectRatio: false,
              animation: false,
              interaction: {{
                mode: 'index',
                intersect: false
              }},
              plugins: {{
                legend: {{
                  display: true
                }},
                tooltip: {{
                  enabled: true
                }}
              }},
              scales: {{
                x: {{
                  ticks: {{
                    maxTicksLimit: 12
                  }}
                }},
                y: {{
                  beginAtZero: true,
                  title: {{
                    display: true,
                    text: 'Knots'
                  }}
                }}
              }}
            }}
          }});
        }}

        loadWindChart();
      </script>

      <br><br>
    
    <h2>Today's Raw 5-Min Buckets (Debug)</h2>

    <table>
    <tr>
    <th>Time</th>
    <th>Avg Wind</th>
    <th>Gust</th>
    </tr>

    {''.join(
    f"<tr><td>{r['time']}</td><td>{r['avg']}</td><td>{r['gust']}</td></tr>"
    for r in debug_rows
    )}

    </table>

    </body>
    </html>
    """
    return Response(html, mimetype="text/html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "wind-pi"})


@app.route("/wind", methods=["GET"])
def get_wind():
    return jsonify(LATEST_WIND)


@app.route("/seed", methods=["POST"])
def seed_fake_data():
    """
    Optional helper endpoint to create one fake completed 5-minute bucket.
    """
    import random

    bucket_end = floor_to_5min(now_pacific())
    bucket_start = bucket_end - timedelta(minutes=5)

    avg_wind = round(random.uniform(4, 16), 1)
    gust_wind = round(avg_wind + random.uniform(1, 6), 1)
    direction = random.randint(0, 359)
    sample_count = random.randint(200, 320)

    insert_bucket(
        bucket_start=bucket_start,
        bucket_end=bucket_end,
        avg_wind_knots=avg_wind,
        gust_knots=gust_wind,
        avg_direction_degrees=direction,
        sample_count=sample_count,
    )
    cleanup_old_data()

    return jsonify({
        "status": "ok",
        "bucket_start": bucket_start.isoformat(timespec="seconds"),
        "bucket_end": bucket_end.isoformat(timespec="seconds"),
        "avg_wind_knots": avg_wind,
        "gust_knots": gust_wind,
        "wind_direction_degrees": direction,
        "sample_count": sample_count,
    })


def build_plot_series():
    end = now_pacific()
    start = end - timedelta(hours=24)
    rows = buckets_since(start)

    if not rows:
        return {
            "labels": [],
            "trimmed_mean_knots": [],
            "gust_knots": [],
        }

    times = [
        datetime.fromisoformat(row["bucket_start"]).astimezone(PACIFIC)
        for row in rows
    ]
    trimmed_mean_speeds = [round(row["avg_wind_knots"], 3) for row in rows]
    gust_speeds = [round(row["gust_knots"], 3) for row in rows]

    alpha = 0.65
    smoothed_trimmed_mean_speeds = []
    for i, x in enumerate(trimmed_mean_speeds):
        if i == 0:
            smoothed_trimmed_mean_speeds.append(round(x, 3))
        else:
            y_prev = smoothed_trimmed_mean_speeds[-1]
            y = alpha * x + (1.0 - alpha) * y_prev
            smoothed_trimmed_mean_speeds.append(round(y, 3))

    labels = [t.strftime("%m-%d %H:%M") for t in times]

    return {
        "labels": labels,
        "trimmed_mean_knots": smoothed_trimmed_mean_speeds,
        "gust_knots": gust_speeds,
    }


@app.route("/plot.json")
def plot_json():
    now = now_pacific()

    if (
        _plot_cache["ts"]
        and (now - _plot_cache["ts"]).total_seconds() < 30
        and _plot_cache["json"] is not None
    ):
        return jsonify(_plot_cache["json"])

    data = build_plot_series()
    _plot_cache["ts"] = now
    _plot_cache["json"] = data

    return jsonify(data)


@app.route("/plot.png")
def plot_png():
    now = now_pacific()

    if (
        _plot_cache["ts"]
        and (now - _plot_cache["ts"]).total_seconds() < 30
        and _plot_cache["png"] is not None
    ):
        return Response(_plot_cache["png"], mimetype="image/png")

    data = build_plot_series()

    if not data["labels"]:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No wind data yet", ha="center", va="center", fontsize=16)
        ax.set_axis_off()
    else:
        times = [
            datetime.strptime(label, "%m-%d %H:%M").replace(year=now.year, tzinfo=PACIFIC)
            for label in data["labels"]
        ]

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(times, data["trimmed_mean_knots"], label="Trimmed Mean Wind", linewidth=2.0)
        ax.plot(times, data["gust_knots"], label="Gust Wind", linewidth=1.5, alpha=0.35)

        ax.set_title("Wind - Last 24 Hours (5-Min Trimmed Mean / Max Gust)")
        ax.set_ylabel("Knots")
        ax.set_xlabel("Pacific Time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz=PACIFIC))
        fig.autofmt_xdate()

    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)

    png_bytes = buf.getvalue()
    _plot_cache["ts"] = now
    _plot_cache["png"] = png_bytes

    return Response(png_bytes, mimetype="image/png")


def start_background_threads():
    threading.Thread(target=serial_loop, daemon=True).start()


if __name__ == "__main__":
    init_db()
    start_background_threads()
    app.run(host="0.0.0.0", port=8000)
else:
    init_db()
    start_background_threads()
