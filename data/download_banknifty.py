from kiteconnect import KiteConnect
import pandas as pd
from datetime import datetime, timedelta
import time

# =====================
# ZERODHA CREDENTIALS
# =====================
API_KEY = "btvvpzbqxracvii2"
ACCESS_TOKEN = "wyUx2G01Ll95zxgpK3acnZMA7Nh1fuYF"

# BankNifty Index instrument token
BANKNIFTY_TOKEN = 260105

kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

# ---------------------
# DATE RANGE (2 YEARS)
# ---------------------
END_DATE = datetime.today()
START_DATE = END_DATE - timedelta(days=730)  # 2 years
CHUNK_DAYS = 100

all_data = []

current_start = START_DATE

print("Starting BankNifty 5-min historical download...")

while current_start < END_DATE:
    current_end = min(current_start + timedelta(days=CHUNK_DAYS), END_DATE)

    print(f"Downloading: {current_start.date()} → {current_end.date()}")

    data = kite.historical_data(
        instrument_token=BANKNIFTY_TOKEN,
        from_date=current_start,
        to_date=current_end,
        interval="5minute"
    )

    if data:
        all_data.extend(data)

    current_start = current_end
    time.sleep(0.4)  # rate-limit safety

df = pd.DataFrame(all_data)

df.rename(columns={"date": "datetime"}, inplace=True)
df = df[["datetime", "open", "high", "low", "close", "volume"]]

df.drop_duplicates(subset=["datetime"], inplace=True)
df.sort_values("datetime", inplace=True)

df.to_csv(
    "data/historical/banknifty_5m.csv",
    index=False
)

print("✅ DOWNLOAD COMPLETE")
print("Rows:", len(df))
print(df.head())
print(df.tail())