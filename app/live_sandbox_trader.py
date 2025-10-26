# live_sandbox_trader.py
# deps: python-dotenv pandas numpy torch torchvision tinkoff-investments joblib pillow
import os, time, json, uuid, math
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
from dotenv import load_dotenv

import torch
import torch.nn as nn

from tinkoff.invest import (
    Client, InstrumentIdType, OrderDirection, OrderType,
    MoneyValue, Quotation
)

# ========= CONFIG =========
load_dotenv()

TINKOFF_TOKEN = os.getenv("TINKOFF_SANDBOX_TOKEN")

TICKER   = os.getenv("TICKER", "SBER")
CLASS    = os.getenv("CLASS_CODE", "TQBR")

TF       = os.getenv("TF", "5m")
H        = int(os.getenv("H", "36"))
TAU      = float(os.getenv("TAU", "0.02"))  # для матмодели

FEAT_TODAY = os.getenv("FEAT_TODAY", f"../data/features/ml/{TICKER}/{TF}/features_today.parquet")
CV_MANIFEST= os.getenv("CV_MANIFEST", f"../data/cv/images/{TICKER}/{TF}/today_manifest.csv")
ML_PKL     = os.getenv("ML_PKL",      f"../models/{TICKER}_{TF}_lgbm.pkl")
CV_MODEL   = os.getenv("CV_MODEL",    f"../models/cv_resnet18_state.pt")
DQN_WEIGHTS= os.getenv("DQN_WEIGHTS", f"../models/rl_{TICKER}_{TF}_dqn.pt")

TOPUP_RUB  = float(os.getenv("TOPUP_RUB", "100000"))
LOTS       = int(os.getenv("LOTS", "1"))
SLEEP_SEC  = int(os.getenv("SLEEP_SEC", "30"))
END_HOUR_MSK = int(os.getenv("END_HOUR_MSK", "23"))

SL_PCT     = float(os.getenv("SL_PCT", "0.01"))
TP_PCT     = float(os.getenv("TP_PCT", "0.02"))
TRAIL_PCT  = float(os.getenv("TRAIL_PCT", "0.0"))
STATE_FILE = os.getenv("STATE_FILE", "state_sandbox.json")

def tf_to_minutes(tf: str) -> int:
    tf = tf.strip().lower()
    if tf.endswith("m"): return int(tf[:-1])
    if tf.endswith("h"): return int(tf[:-1]) * 60
    if tf.endswith("d"): return int(tf[:-1]) * 60 * 24
    raise ValueError(f"unsupported TF: {tf}")

TF_MIN = tf_to_minutes(TF)

def money_rub(amount: float) -> MoneyValue:
    units = int(math.floor(amount))
    nano = int(round((amount - units) * 1e9))
    return MoneyValue(currency="rub", units=units, nano=nano)

def to_quotation(price: float) -> Quotation:
    units = int(math.floor(price))
    nano = int(round((price - units) * 1e9))
    return Quotation(units=units, nano=nano)

def now_utc():
    return datetime.now(timezone.utc)

def msk_now():
    return now_utc() + timedelta(hours=3)

def load_state():
    p = Path(STATE_FILE)
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"positions": {}}

def save_state(state: dict):
    Path(STATE_FILE).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

# === Inference hooks ===
# ВАЖНО: используем infer_math_from_prices, а не infer_math
from RL.inference import infer_ml, infer_cv, infer_math_from_prices

def get_signals_for_now() -> dict:
    ml, cv, mt = {}, {}, {}
    # ML: порядок аргументов (model_path, feat_path, H)
    try:
        ml, _ = infer_ml(ML_PKL, FEAT_TODAY, H)
        ml = ml or {}
    except Exception as e:
        print("[warn] ML infer:", e)
    # CV: агрегированный score по последним кадрам
    try:
        if Path(CV_MODEL).exists() and Path(CV_MANIFEST).exists():
            cv_out, st = infer_cv(CV_MODEL, CV_MANIFEST)
            if st == "ok" and isinstance(cv_out, dict):
                cv = cv_out
    except Exception as e:
        print("[warn] CV infer:", e)
    # Math: используем путь к parquet и H*TF_MIN минут
    try:
        if Path(FEAT_TODAY).exists():
            mt, _ = infer_math_from_prices(FEAT_TODAY, H_min=H*TF_MIN, thr_pct=TAU)
            mt = mt or {}
    except Exception as e:
        print("[warn] MATH infer:", e)

    s_ml = float(ml.get("score", np.nan))
    s_cv = float(cv.get("score", np.nan))
    p_up = float(ml.get("P_up", np.nan))
    p_dn = float(ml.get("P_down", np.nan))
    p_tau= float(mt.get("p_up_tau", np.nan))
    s_ens = np.nanmean([s_ml, s_cv])
    feats = dict(s_ml=s_ml, s_cv=s_cv, s_ens=s_ens, p_up=p_up, p_down=p_dn, p_up_tau=p_tau)
    for k,v in feats.items():
        if not math.isfinite(v):
            feats[k] = 0.0
    return feats

# === RL agent ===
class QNet(nn.Module):
    def __init__(self, n_obs=6, n_act=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_act)
        )
    def forward(self, x): return self.net(x)

def dqn_decide(features: dict) -> int:
    feats_order = ["s_ml","s_cv","s_ens","p_up","p_down","p_up_tau"]
    x = np.array([features.get(k, 0.0) for k in feats_order], dtype=np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q = QNet(n_obs=len(feats_order), n_act=3).to(device)
    sd = torch.load(DQN_WEIGHTS, map_location=device, weights_only=True)
    q.load_state_dict(sd, strict=True)
    q.eval()
    with torch.no_grad():
        qv = q(torch.from_numpy(x).to(device).unsqueeze(0))
        a_idx = int(torch.argmax(qv, dim=1).item())
    return a_idx - 1  # {-1,0,1}

# === Tinkoff Sandbox ===
def ensure_sandbox_account(cli: Client) -> str:
    # Deprecated warning допустим. Можно заменить на users.get_accounts для чтения.
    accs = cli.sandbox.get_sandbox_accounts().accounts
    if accs:
        return accs[0].id
    opened = cli.sandbox.open_sandbox_account()
    account_id = opened.account_id
    # верный метод пополнения песочницы
    cli.sandbox.sandbox_pay_in(account_id=account_id, amount=money_rub(TOPUP_RUB))
    return account_id

def resolve_figi(cli: Client, ticker: str, class_code: str="TQBR") -> str:
    r = cli.instruments.share_by(
        id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_TICKER,
        id=ticker,
        class_code=class_code,
    )
    return r.instrument.figi

def get_positions(cli: Client, account_id: str) -> dict:
    pos = cli.sandbox.get_sandbox_positions(account_id=account_id)
    return {p.figi: p.quantity.lots for p in pos.securities}

def get_last_price(cli: Client, figi: str) -> float:
    r = cli.market_data.get_last_prices(figi=[figi])
    if not r.last_prices:
        return float("nan")
    q = r.last_prices[0].price
    return float(q.units + q.nano/1e9)

def place_market(cli: Client, account_id: str, figi: str, lots: int, side: str):
    direction = OrderDirection.ORDER_DIRECTION_BUY if side == "BUY" else OrderDirection.ORDER_DIRECTION_SELL
    oid = str(uuid.uuid4())
    resp = cli.sandbox.post_sandbox_order(
        order_id=oid,
        account_id=account_id,
        figi=figi,
        quantity=int(abs(lots)),
        direction=direction,
        order_type=OrderType.ORDER_TYPE_MARKET,
    )
    return {"order_id": oid, "status": str(resp.execution_report_status), "lots": lots, "side": side}

def enforce_long_only(action: int, current_lots: int) -> int:
    target = LOTS if action > 0 else 0
    return int(target - max(current_lots, 0))

def flatten_before_close(cli: Client, account_id: str, figi: str):
    lots_now = int(get_positions(cli, account_id).get(figi, 0))
    if lots_now > 0:
        place_market(cli, account_id, figi, lots_now, side="SELL")

# === MAIN LOOP ===
def main():
    if not TINKOFF_TOKEN:
        raise RuntimeError("Нет TINKOFF_SANDBOX_TOKEN в .env")

    state = load_state()

    with Client(TINKOFF_TOKEN) as cli:
        account_id = ensure_sandbox_account(cli)
        figi = resolve_figi(cli, TICKER, CLASS)
        print(json.dumps({"account_id": account_id, "figi": figi, "ticker": TICKER}, ensure_ascii=False))

        while True:
            try:
                if msk_now().hour >= END_HOUR_MSK:
                    try:
                        flatten_before_close(cli, account_id, figi)
                        state["positions"].pop(figi, None); save_state(state)
                    except Exception as e:
                        print("[warn] flatten:", e)
                    time.sleep(SLEEP_SEC); continue

                feats = get_signals_for_now()
                action = dqn_decide(feats)

                lots_map = get_positions(cli, account_id)
                cur_lots = int(lots_map.get(figi, 0))
                delta = enforce_long_only(action, cur_lots)

                info = {
                    "time": datetime.utcnow().isoformat()+"Z",
                    "features": feats,
                    "action": int(action),
                    "current_lots": cur_lots,
                    "delta": int(delta)
                }

                if delta > 0:
                    res = place_market(cli, account_id, figi, delta, side="BUY")
                    info["order"] = res

                    px = get_last_price(cli, figi)
                    if not math.isfinite(px):
                        try:
                            df = pd.read_parquet(FEAT_TODAY)
                            px = float(df["close"].iloc[-1])
                        except Exception:
                            px = 0.0
                    entry_lots = cur_lots + delta
                    state["positions"][figi] = {
                        "lots": int(entry_lots),
                        "entry_px": px,
                        "stop_px":  px*(1.0 - SL_PCT) if SL_PCT>0 else None,
                        "tp_px":    px*(1.0 + TP_PCT) if TP_PCT>0 else None,
                        "trail_high": px,
                        "trail_pct": TRAIL_PCT
                    }
                    save_state(state)

                elif delta < 0:
                    res = place_market(cli, account_id, figi, abs(delta), side="SELL")
                    info["order"] = res
                    lots_left = cur_lots + delta
                    if lots_left <= 0:
                        state["positions"].pop(figi, None)
                    else:
                        pos = state["positions"].get(figi, {})
                        pos["lots"] = int(lots_left)
                        state["positions"][figi] = pos
                    save_state(state)
                else:
                    info["order"] = None

                print(json.dumps(info, ensure_ascii=False))

                # SL/TP + трейлинг
                try:
                    lots_now = int(get_positions(cli, account_id).get(figi, 0))
                    if lots_now > 0 and figi in state["positions"]:
                        pos = state["positions"][figi]
                        px = get_last_price(cli, figi)
                        if math.isfinite(px):
                            tr_pct = float(pos.get("trail_pct", 0.0) or 0.0)
                            if tr_pct > 0.0:
                                pos["trail_high"] = max(float(pos.get("trail_high", px)), px)
                                trail_stop = pos["trail_high"] * (1.0 - tr_pct)
                                if pos.get("stop_px") is None:
                                    pos["stop_px"] = trail_stop
                                else:
                                    pos["stop_px"] = max(float(pos["stop_px"]), trail_stop)

                            hit_sl = (pos.get("stop_px") is not None) and (px <= float(pos["stop_px"]))
                            hit_tp = (pos.get("tp_px")   is not None) and (px >= float(pos["tp_px"]))

                            if hit_sl or hit_tp:
                                place_market(cli, account_id, figi, lots_now, side="SELL")
                                state["positions"].pop(figi, None); save_state(state)
                                print(json.dumps({"event":"exit", "reason":"SL" if hit_sl else "TP",
                                                  "px":px, "lots":lots_now}, ensure_ascii=False))
                except Exception as e:
                    print("[warn] sl/tp check:", e)

            except Exception as e:
                print("[err] loop:", e)

            time.sleep(SLEEP_SEC)

if __name__ == "__main__":
    main()
