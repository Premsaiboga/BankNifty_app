from backtest.backtest_engine import BacktestEngine

engine = BacktestEngine(rr_list=[2, 3, 4])

engine.run(
    csv_path="data/historical/banknifty_5m.csv",
    output_path="data/historical/trade_log.csv"
)
