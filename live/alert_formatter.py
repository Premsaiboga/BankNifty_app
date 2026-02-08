def format_trade_alert(trade, prob):
    return f"""
📌 *BANKNIFTY TRADE ALERT*

*Strategy* : {trade["strategy"]}
*Type*     : {trade["type"]}
*Entry*    : {trade["entry"]}
*SL*       : {trade["stoploss"]}
*Target*   : {trade["target"]} (RR 1:{trade["rr"]})
*AI Prob*  : {round(prob, 2)}
*Time*     : {trade["time"]}
""".strip()
