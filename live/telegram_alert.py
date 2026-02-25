import requests
import os
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def send_telegram_alert(message: str):

    if not BOT_TOKEN or not CHAT_ID:
        print("⚠ Telegram credentials missing")
        return

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    payload = {
        "chat_id": CHAT_ID,
        "text": message
    }

    try:
        response = requests.post(url, json=payload, timeout=5)

        if response.status_code == 200:
            print("✅ Telegram alert sent")
        else:
            print(f"Telegram error: {response.status_code} - {response.text}")

    except Exception as e:
        print("Telegram error:", e)