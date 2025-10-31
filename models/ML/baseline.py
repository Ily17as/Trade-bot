import numpy as np, pandas as pd, lightgbm as lgb
import joblib

FEAT_LBL, H = "../data/features/ml/SBER/5m/features_labeled.parquet", 36
df = pd.read_parquet(FEAT_LBL).sort_values("time")

# 1) метка
y_raw = df.get(f"y_tb_{H}", df.get(f"y_tb_ft_{H}"))
if y_raw is None: raise KeyError("нет колонки метки")
y = (pd.to_numeric(y_raw.replace([np.inf,-np.inf], np.nan), errors="coerce")
       .dropna()
       .astype(np.int8))
df = df.loc[y.index].reset_index(drop=True)   # синхронизируем

# 2) матрица признаков: только числовые/булевы
drop_cols = [c for c in df.columns if c.startswith(("y_tb_","ret_fwd_"))] + ["time"]
X = df.drop(columns=drop_cols, errors="ignore")

# удалим object/строки (например, 'date') или закодируй при желании
obj_cols = X.select_dtypes(include=["object"]).columns.tolist()
if obj_cols:
    print("drop object cols:", obj_cols)
    X = X.drop(columns=obj_cols)

# привести bool→uint8, ffill + нули
for c in X.select_dtypes(include=["bool"]).columns: X[c] = X[c].astype(np.uint8)
X = (X.replace([np.inf,-np.inf], np.nan)
       .ffill()
       .fillna(0.0))

print("label dist:", y.value_counts(dropna=False).to_dict())

y2 = (y+1).astype(int)                          # {0,1,2}

# class weights
freq = y2.value_counts(normalize=True)
wmap = {k:1/max(1e-9,freq.get(k,1e-9)) for k in [0,1,2]}
w = y2.map(wmap)

# простой временной сплит: train < последний месяц, val = последний месяц
ts = pd.to_datetime(df["time"])
cut = ts.max() - pd.Timedelta(days=30)
tr, va = ts < cut, ts >= cut

dtr = lgb.Dataset(X[tr], label=y2[tr], weight=w[tr])
params = dict(objective="multiclass", num_class=3, learning_rate=0.05,
              num_leaves=63, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, min_data_in_leaf=100, metric="multi_logloss")
m = lgb.train(params, dtr, num_boost_round=800)
proba = m.predict(X[va])                         # shape [N,3] порядок классов {0,1,2}={down,flat,up}
p_dn, p_fl, p_up = proba[:,0], proba[:,1], proba[:,2]
score = p_up - p_dn

joblib.dump(m, "../models/SBER_5m_lgbm.pkl")