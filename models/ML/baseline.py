# baseline_xgb.py
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb

# Путь к файлу (на твоей машине)
PATH = "../../data/SBER_dataset_5m.csv"

df = pd.read_csv(PATH, parse_dates=['time'], infer_datetime_format=True)
# Подставь правильное имя столбца времени
if 'time' in df.columns:
    df = df.sort_values('time').reset_index(drop=True)
else:
    df = df.sort_values(df.columns[0]).reset_index(drop=True)

# Простые признаки
df['close'] = df['close'].astype(float)
df['logret'] = np.log(df['close']).diff()
df['ret_1'] = df['logret'].shift(1)
for w in [3,5,10,20]:
    df[f'sma_{w}'] = df['close'].rolling(window=w).mean()
    df[f'std_{w}'] = df['logret'].rolling(window=w).std()

# ATR (упрощённо)
df['high'] = df['high'].astype(float)
df['low']  = df['low'].astype(float)
df['tr1'] = (df['high'] - df['low'])
df['tr2'] = (df['high'] - df['close'].shift(1)).abs()
df['tr3'] = (df['low'] - df['close'].shift(1)).abs()
df['tr'] = df[['tr1','tr2','tr3']].max(axis=1)
df['atr_14'] = df['tr'].rolling(14).mean()

# Label: triple-barrier simplified -> use next k returns and ATR thresholds
H = 3  # horizon steps (3*5min = 15min)
tp_atr = 0.5  # take profit multiplier
sl_atr = 0.5  # stoploss multiplier

df['future_max'] = df['close'].shift(-1).rolling(window=H).max().shift(-(H-1))
df['future_min'] = df['close'].shift(-1).rolling(window=H).min().shift(-(H-1))
df['target'] = 0  # 1 up, -1 down, 0 flat

# dynamic barriers
df['tp'] = df['close'] + tp_atr * df['atr_14']
df['sl'] = df['close'] - sl_atr * df['atr_14']

# If future_max crosses tp => up, elif future_min crosses sl => down, else flat
df.loc[df['future_max'] >= df['tp'], 'target'] = 1
df.loc[df['future_min'] <= df['sl'], 'target'] = -1

# Dropna and features
feats = ['ret_1', 'sma_3','sma_5','sma_10','std_3','std_5','std_10','atr_14']
data = df.dropna(subset=feats + ['target']).reset_index(drop=True)
X = data[feats]
y = data['target'].map({-1:0, 0:1, 1:2})  # map to 3 classes for classifier

# Time split (train last)
tscv = TimeSeriesSplit(n_splits=5)
split = list(tscv.split(X))
train_idx, test_idx = split[-1]  # last fold
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
params = {
    'objective':'multi:softprob',
    'num_class':3,
    'eval_metric':'mlogloss',
    'tree_method':'hist',
    'learning_rate':0.05,
    'max_depth':6,
    'seed':42
}
bst = xgb.train(params, dtrain, num_boost_round=200,
                evals=[(dtrain,'train'), (dtest,'test')], early_stopping_rounds=20, verbose_eval=False)

y_pred = np.argmax(bst.predict(dtest), axis=1)
print(classification_report(y_test, y_pred, digits=4))
print(confusion_matrix(y_test, y_pred))


bst.save_model("xgb_sber.model")        # бинарный формат XGBoost
bst.save_model("xgb_sber.json")         # json формат (читаемый)


# Загрузка
bst2 = xgb.Booster()
bst2.load_model("xgb_sber.model")
# Для предсказания на numpy/pandas:
dtest = xgb.DMatrix(X_test)   # X_test — DataFrame/np.array
probs = bst2.predict(dtest)