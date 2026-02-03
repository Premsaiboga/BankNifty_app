from kiteconnect import KiteConnect

API_KEY = "btvvpzbqxracvii2"
API_SECRET = "k4k75n7itt0fy6iun5l53nq8l4bho76f"
REQUEST_TOKEN = "w3i1vPui2jfmOBCllopcBhc4V9P1wXRB"

kite = KiteConnect(api_key=API_KEY)

data = kite.generate_session(
    REQUEST_TOKEN,
    api_secret=API_SECRET
)

print("ACCESS TOKEN:", data["access_token"])
