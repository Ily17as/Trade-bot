# airflow/scripts/update_dataset.py
import os
from datetime import datetime, timedelta, timezone


def get_interval(tf: str):
    from tinkoff.invest import CandleInterval

    mapping = {
        "1m": CandleInterval.CANDLE_INTERVAL_1_MIN,
        "5m": CandleInterval.CANDLE_INTERVAL_5_MIN,
        "15m": CandleInterval.CANDLE_INTERVAL_15_MIN,
        "1h": CandleInterval.CANDLE_INTERVAL_HOUR,
        "1d": CandleInterval.CANDLE_INTERVAL_DAY,
    }
    # по умолчанию 5m
    return mapping.get(tf, CandleInterval.CANDLE_INTERVAL_5_MIN)


def update_dataset():
    import pandas as pd
    from tinkoff.invest import Client

    token = os.getenv("TINKOFF_TOKEN")
    if not token:
        raise RuntimeError("TINKOFF_TOKEN is not set in environment/.env")

    project_path = os.getenv("PROJECT_PATH", "/opt/airflow/Trade-bot")
    save_path = os.path.join(project_path, "data", "latest.csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    ticker = os.getenv("TICKER", "SBER")
    tf = os.getenv("TF", "5m")

    # последние 30 дней в UTC
    now_utc = datetime.now(timezone.utc)
    from_dt = now_utc - timedelta(days=30)
    to_dt = now_utc

    with Client(token) as client:
        # ищем FIGI по тикеру
        shares = client.instruments.shares().instruments
        figi = next((s.figi for s in shares if s.ticker == ticker), None)
        if figi is None:
            raise ValueError(f"FIGI for {ticker} not found")

        interval = get_interval(tf)

        candles = client.get_all_candles(
            figi=figi,
            from_=from_dt,
            to=to_dt,
            interval=interval,
        )

        rows = []
        for c in candles:
            rows.append(
                {
                    "time": c.time,  # уже tz-aware UTC
                    "open": c.open.units + c.open.nano / 1e9,
                    "high": c.high.units + c.high.nano / 1e9,
                    "low": c.low.units + c.low.nano / 1e9,
                    "close": c.close.units + c.close.nano / 1e9,
                    "volume": c.volume,
                }
            )

        df = pd.DataFrame(rows)
        df.to_csv(save_path, index=False)

    print("Dataset updated:", save_path)
