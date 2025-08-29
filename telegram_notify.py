import os, time, requests

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID   = os.environ["TELEGRAM_CHAT_ID"]
API = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

def send_message(text, parse_mode="HTML"):
    payload = {
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }
    for i in range(3):
        try:
            r = requests.post(API, json=payload, timeout=20)
            r.raise_for_status()
            return
        except Exception:
            if i == 2:
                raise
            time.sleep(2 * (i + 1))
      
