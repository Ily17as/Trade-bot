## Model Description
**"best_ssm_model.pth"** is a recurrent neural network that analyzes sequences of stock market data including prices, technical indicators, and past trend labels. It predicts whether the price will go up, stay flat, or go down in the next period.

## Input data
The same type of data that we have in the file **"SBER_dataset_5m.csv"**.

The file with the new data must be in the format .csv and contain all the columns from the list:

    'open', 
    'high', 
    'low', 
    'close', 
    'volume',
    'sma_5', 
    'sma_10', 
    'ema_10', 
    'ema_20', 
    'sma_ratio',
    'rsi', 
    'boll_mid', 
    'boll_std', 
    'boll_up', 
    'boll_low', 
    'boll_pos',
    'momentum_5', 
    'momentum_10', 
    'log_ret', 
    'atr'

## Usage
You can run **"ssm_model_for_new_data.py"** or just the code below (if you run it from the current directory).

```python
from ssm_model import predict_new_data

model_path = "best_ssm_model.pth"
new_data_path = "new_data.csv"  # for example
predict_new_data(model_path, new_data_path).to_csv('predicted_data.csv', index=False)

```
In the created file, the "predicted_label" column contains the values up/flat/down for each record.
