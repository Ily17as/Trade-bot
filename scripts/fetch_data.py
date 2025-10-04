# --- PATCH START ---

import pandas as pd, requests, yfinance as yf
from datetime import datetime, timedelta, timezone, date
from dateutil.relativedelta import relativedelta
from tinkoff.invest import Client, CandleInterval
from scripts.env_tools import getenv
from scripts.paths import partdir
import argparse

INTERVAL_MAP = {
    "1m": CandleInterval.CANDLE_INTERVAL_1_MIN,
    "5m": CandleInterval.CANDLE_INTERVAL_5_MIN,
    "1h": CandleInterval.CANDLE_INTERVAL_HOUR,
    "1d": CandleInterval.CANDLE_INTERVAL_DAY,
}

# допустимая ширина окна запроса под каждый TF (эмпирически безопасно)
TF_WINDOW = {
    "1m": timedelta(days=1),     # 1 день за запрос
    "5m": timedelta(days=7),     # ~неделя
    "1h": relativedelta(months=1),
    "1d": relativedelta(years=1),
}

def _money(v): return v.units + v.nano/1e9

def _dt(obj):
    return obj if isinstance(obj, datetime) else datetime.fromisoformat(str(obj))

def daterange_chunks(start_dt: datetime, end_dt: datetime, step):
    cur = start_dt
    while cur < end_dt:
        if isinstance(step, relativedelta):
            nxt = cur + step
        else:
            nxt = cur + step
        yield cur, min(nxt, end_dt)
        cur = nxt

def fetch_tinkoff_chunked(figi: str, tf: str, start: datetime, end: datetime) -> pd.DataFrame:
    token = getenv("TINKOFF_TOKEN")
    if not token:
        return pd.DataFrame()
    rows = []
    with Client(token) as c:
        for left, right in daterange_chunks(start, end, TF_WINDOW[tf]):
            resp = c.market_data.get_candles(
                figi=figi,
                from_=left.replace(tzinfo=timezone.utc),
                to=right.replace(tzinfo=timezone.utc),
                interval=INTERVAL_MAP[tf],
            )
            rows.extend({
                "time": x.time, "open": _money(x.open), "high": _money(x.high),
                "low": _money(x.low), "close": _money(x.close), "volume": x.volume
            } for x in resp.candles)
    return pd.DataFrame(rows)

def fetch_moex_history(secid: str, tf: str, date_from: str, date_till: str) -> pd.DataFrame:
    # MOEX поддерживает интервалы: 1,10,60,1440
    tf_map = {"1m": 1, "5m": 10, "1h": 60, "1d": 1440}
    interval = tf_map[tf]
    base = "https://iss.moex.com/iss/history/engines/stock/markets/shares/boards/TQBR/securities/{sec}/candles.json"
    url = base.format(sec=secid)
    start = 0
    out = []
    while True:
        p = {"from": date_from, "till": date_till, "interval": interval, "start": start}
        r = requests.get(url, params=p, timeout=30)
        r.raise_for_status()
        js = r.json()
        # аккуратная проверка структуры
        if "candles" not in js or "data" not in js["candles"] or not js["candles"]["data"]:
            break
        cols = js["candles"]["columns"]; data = js["candles"]["data"]
        df = pd.DataFrame(data, columns=cols)
        out.append(df); start += len(df)
    if not out:
        return pd.DataFrame()
    df = pd.concat(out, ignore_index=True)
    if "begin" in df.columns:
        df = df.rename(columns={"begin": "time"})
    df["time"] = pd.to_datetime(df["time"])
    return df[["time","open","high","low","close","volume"]].sort_values("time")

def fetch_yf(symbol, tf, start):
    # минутки у Yahoo доступны только ~за последние 7–8 дней
    df = yf.download(symbol, interval=tf, start=start, auto_adjust=False, progress=False)
    if df.empty: return df
    df = df.rename(columns=str.lower).reset_index()
    if "datetime" in df.columns: df = df.rename(columns={"datetime":"time"})
    if "date" in df.columns: df = df.rename(columns={"date":"time"})
    return df[["time","open","high","low","close","volume"]]

def save_partitioned(df, source, ticker, tf):
    if df.empty: return
    df = df.drop_duplicates("time").sort_values("time")
    dt = pd.to_datetime(df["time"], utc=True)
    df["year"] = dt.dt.year; df["month"] = dt.dt.month
    for (y,m), chunk in df.groupby(["year","month"]):
        out = partdir(source, ticker, tf, int(y), int(m)) / "part.parquet"
        chunk.drop(columns=["year","month"]).to_parquet(out, index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figi", help="FIGI для Tinkoff", required=False)
    ap.add_argument("--ticker", required=True, help="MOEX тикер, напр. SBER")
    ap.add_argument("--tf", default="1m", choices=["1m","5m","1h","1d"])
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default=datetime.utcnow().date().isoformat())
    ap.add_argument("--yf_symbol", default=None)
    args = ap.parse_args()

    START = _dt(args.start); END = _dt(args.end)
    ok = False

    # 1) Tinkoff (chunked)
    try:
        if args.figi and getenv("TINKOFF_TOKEN"):
            dft = fetch_tinkoff_chunked(args.figi, args.tf, START, END)
            if not dft.empty:
                save_partitioned(dft, "tinkoff", args.ticker, args.tf); ok = True
    except Exception as e:
        print("Tinkoff fail:", e)

    # 2) MOEX
    if not ok:
        try:
            dfm = fetch_moex_history(args.ticker, args.tf, args.start, args.end)
            if not dfm.empty:
                save_partitioned(dfm, "moex", args.ticker, args.tf); ok = True
        except Exception as e:
            print("MOEX fail:", e)

    # 3) yfinance (ограничен по 1m)
    if not ok:
        symbol = args.yf_symbol or f"{args.ticker}.ME"
        dfy = fetch_yf(symbol, args.tf, args.start)
        if not dfy.empty:
            save_partitioned(dfy, "yfinance", args.ticker, args.tf); ok = True

    if not ok:
        raise SystemExit("Нет данных ни из одного источника.")

if __name__ == "__main__": main()

# --- PATCH END ---
