from kiteconnect import KiteConnect
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# =========================
# LOAD ENV
# =========================
load_dotenv()

API_KEY = os.getenv("API_KEY")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")

kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

# =========================
# CONFIG
# =========================
INSTRUMENT_TOKEN = 260105  # BANKNIFTY spot
INTERVAL = "minute"

START_DATE = datetime(2024, 2, 1)
END_DATE   = datetime(2026, 2, 5)

SAVE_PATH = "data/historical/banknifty_1min.csv"

# =========================
# FETCH DATA (60-day chunks)
# =========================
all_data = []

from_date = START_DATE

while from_date < END_DATE:
    to_date = min(from_date + timedelta(days=60), END_DATE)

    print(f"Fetching {from_date.date()} → {to_date.date()}")

    candles = kite.historical_data(
        instrument_token=INSTRUMENT_TOKEN,
        from_date=from_date,
        to_date=to_date,
        interval=INTERVAL
    )

    if candles:
        df = pd.DataFrame(candles)
        all_data.append(df)

    from_date = to_date + timedelta(days=1)

# =========================
# SAVE CSV
# =========================
final_df = pd.concat(all_data, ignore_index=True)

final_df.rename(columns={"date": "time"}, inplace=True)
final_df["time"] = pd.to_datetime(final_df["time"])

final_df.to_csv(SAVE_PATH, index=False)

print(f"\n✅ Saved {len(final_df)} rows to {SAVE_PATH}")
