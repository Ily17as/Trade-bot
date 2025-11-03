from inference_pipeline import run_full_inference

results = run_full_inference(
    ticker="SBER",              # тикер
    days=2,                     # за сколько последних дней взять данные
    ml_model_path="xgb_sber.model",
    cv_model_path="best_model.pth",
    meta_path="meta.json",
    lookback=60                 # длина окна свечей для картинки
)

print(results)