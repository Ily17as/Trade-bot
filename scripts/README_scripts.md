# TL;DR
`/scripts` — это набор утилит под датасеты. Правильный путь — запускать **ноутбук `build_datasets.ipynb`**, а скрипты дергать при необходимости. Ниже кратко: что делает каждый файл и как запускать.

---

# README для `scripts/`

## Как пользоваться правильно
1) Открой `build_datasets.ipynb` в корне проекта.  
2) Заполни параметры (`TICKER`, `TF`, `START`, `END`, `FIGI`, `YF_SYMBOL`).  
3) Запусти ячейки сверху вниз: загрузка → фичи → метки → картинки (опционально) → QC → витрина/просмотр.  
4) Скрипты эта тетрадь вызывает за тебя. Вручную они нужны редко.

## Требования
- Python 3.10+
- `pip install -r requirements.txt`
- Файл `.env` в корне:
  ```
  TINKOFF_TOKEN=ваш_токен
  ```

## Состав скриптов

### `env_tools.py`
- **Задача:** загрузить `.env` и отдавать переменные окружения.
- **API:** 
  ```python
  from scripts.env_tools import getenv
  TOKEN = getenv("TINKOFF_TOKEN")
  ```

### `paths.py`
- **Задача:** структура путей и создание директорий (`data/raw`, `data/features`, `data/cv`, ...).
- **API:** `partdir(source, ticker, tf, year, month)` — путь к партиции.

### `fetch_data.py`
- **Задача:** исторические свечи (1m/5m/1h/1d) из **Tinkoff → MOEX → yfinance** с ограничениями по окнам; сохранение в **Parquet** по партициям.
- **Основные функции:** `fetch_tinkoff_chunked`, `fetch_moex_history`, `fetch_yf`, `save_partitioned`. В ноутбуке есть обёртка `fetch_intraday(...)` (либо добавлена в модуле).
- **CLI пример:**
  ```powershell
  python -m scripts.fetch_data --figi BBG004730N88 --ticker SBER --tf 1m --start 2025-09-20 --end 2025-10-04 --yf_symbol SBER.ME
  ```
  Данные лягут в `data/raw/<source>/SBER/1m/year=YYYY/month=MM/part.parquet`.

### `make_intraday_features.py`
- **Задача:** интрадей-фичи из сырых партиций: log-returns, волатильность, VWAP, расстояние до VWAP, индикаторы времени внутри дня.
- **Выход:** `data/features/ml/{TICKER}/{TF}/features.parquet`
- **CLI:**
  ```powershell
  python -m scripts.make_intraday_features --source tinkoff --ticker SBER --tf 1m
  ```

### `make_labels.py`
- **Задача:** метки для ML: `ret_fwd_k` и `y_tb_k` (triple barrier).
- **Вход:** features.parquet → **Выход:** features_labeled.parquet.
- **CLI:**
  ```powershell
  python -m scripts.make_labels --in_parquet data/features/ml/SBER/1m/features.parquet `
                                --out_parquet data/features/ml/SBER/1m/features_labeled.parquet `
                                --k 20
  ```

### `render_cv_images.py`
- **Задача:** рендер свечных **изображений окон** для CV. По умолчанию — без осей; можно включить нижнюю панель **объёмов** и сохранить **манифест CSV**.
- **Параметры:** `size`, `step`, `horizon`, `show_axes`, `show_volume`, `vol_mode={'raw','zscore','rel'}`, `manifest_csv`.
- **CLI:**
  ```powershell
  python -m scripts.render_cv_images --in_parquet data/features/ml/SBER/1m/features_labeled.parquet `
                                     --ticker SBER --tf 1m --size 64 --step 16 --horizon 5
  ```
  Выход: `data/cv/images/{TICKER}/{TF}/win{size}_step{step}/{label}/img_*.png` и (опц.) `manifest.csv`.

### `qc_report.py`
- **Задача:** QC сырых партиций: дубликаты, отрицательные цены, нулевые объёмы, максимальные гепы по времени.
- **API:**
  ```python
  from scripts.qc_report import qc_summary
  qc_summary('data/raw/*/SBER/1m/year=*/month=*/part.parquet')
  ```

### Дополнительно (если присутствуют)
- `loaders.py` — удобные загрузчики партиций → DataFrame.
- `features_intraday.py` / `sessionize.py` — расширенные фичи и признаки сессий.
- `backtest_minute.py`, `strategy_vwap_mr.py` — примеры стратегии и бэктеста без овернайта.
- `build_mart_duckdb.py` — витрины в DuckDB (не обязателен; можно собрать единый Parquet).

## Типовой сценарий в ноутбуке (`build_datasets.ipynb`)
1) **Параметры**: `TICKER`, `TF`, даты, `FIGI`, `YF_SYMBOL`  
2) **Загрузка**: `fetch_intraday(...)` → `save_partitioned(...)`  
3) **Фичи**: `make_intraday_features.build_features(glob)` → `features.parquet`  
4) **Метки**: `make_labels.next_k_return` + `triple_barrier` → `features_labeled.parquet`  
5) **CV-картинки**: `render_with_manifest(..., show_volume=True, manifest_csv=...)`  
6) **QC**: `qc_summary(glob)`  
7) **Витрина без DuckDB**: склейка `features*.parquet` → `warehouse/{ticker}_{tf}.parquet`  
8) **Просмотр**: `pd.read_parquet(...).tail()` и предпросмотр PNG

## FAQ
- **DuckDB обязателен?** Нет. Для простоты склеивай Parquet и работай через pandas/Polars.  
- **Почему картинки без осей?** Меньше утечек признаков, чище вход для CV. Для дебага включай `show_axes=True`.  
- **Где объёмы?** В `render_cv_images.py` параметр `show_volume=True` и режимы нормализации `vol_mode`.
