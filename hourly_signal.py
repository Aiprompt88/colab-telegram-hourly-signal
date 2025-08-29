import os, traceback
from signal_logic import compute_top_signals
from telegram_notify import send_message

def env(name, default=None):
    v = os.environ.get(name)
    return v if (v is not None and v != "") else default

def main():
    try:
        res = compute_top_signals(
            quote_ccy       = env("QUOTE_CCY", "USDT"),
            timeframe       = env("TIMEFRAME", "1h"),
            lookback        = int(env("LOOKBACK", "900")),
            horizon         = int(env("HORIZON", "12")),
            vol_min         = float(env("VOL_MIN", "2000000")),
            vol_max         = float(env("VOL_MAX", "5000000000")),
            scan_all        = env("SCAN_ALL", "false").lower() == "true",
            top_n           = int(env("TOP_N", "200")),
            tp_pct          = env("TP_PCT", "5%"),
            sl_pct          = env("SL_PCT", "3%"),
            fee_pct         = env("FEE_PCT", "0.06%"),
            prob_floor      = float(env("PROB_FLOOR", "0.60")),
            require_trend   = env("REQUIRE_TREND", "true").lower() == "true",
            use_vol         = env("USE_VOL", "true").lower() == "true",
            vol_thresh      = env("VOL_THRESH", "1.5%"),
            tie_policy      = env("TIE_POLICY", "sl_first"),
            short_history   = env("SHORT_HISTORY", "false").lower() == "true",
            max_symbols_in_message = int(env("MAX_ROWS", "8")),
        )
        if res["status"] == "ok":
            send_message(res["message"])
        else:
            # remove this line if you don't want "no signal" notes
            send_message(f"<i>No signal:</i> {res['message']}")
    except Exception as e:
        try:
            send_message(f"<b>Signal job error</b>\n<code>{e}</code>\n\n<code>{traceback.format_exc()[:1500]}</code>")
        except:
            pass
        raise

if __name__ == "__main__":
    main()
          
