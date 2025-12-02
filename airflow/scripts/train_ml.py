def train_ml():
    import joblib
    import pandas as pd
    from xgboost import XGBClassifier
    import os

    data_path = os.getenv("PROJECT_PATH") + "/data/latest.csv"
    df = pd.read_csv(data_path)

    # Пример простых фичей
    df["return"] = df["close"].pct_change()
    df = df.dropna()

    X = df[["open", "close", "high", "low", "volume"]]
    y = (df["return"] > 0).astype(int)

    model = XGBClassifier(n_estimators=50)
    model.fit(X, y)

    save_path = os.getenv("PROJECT_PATH") + "/models/ml/model.pkl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)

    print("ML model saved to:", save_path)
