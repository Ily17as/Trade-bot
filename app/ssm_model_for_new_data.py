import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import math

torch.manual_seed(42)
np.random.seed(42)


class Config:
    input_dim = 12
    d_model = 256
    n_layers = 4
    n_classes = 3
    batch_size = 32
    dropout_rate = 0.1


config = Config()

FEATURE_COLUMNS = [
    'close',
    'volume',
    'sma_ratio',
    'rsi',
    'boll_pos',
    'boll_std',
    'momentum_5',
    'log_ret',
    'atr',
    'label-1', 'label-2', 'label-3'
]

TARGET_COLUMN = 'label'


class FinancialSSM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.input_proj = nn.Linear(config.input_dim, config.d_model)

        self.gru = nn.GRU(
            input_size=config.d_model,
            hidden_size=config.d_model,
            num_layers=config.n_layers,
            batch_first=True,
            dropout=config.dropout_rate if config.n_layers > 1 else 0
        )

        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.classifier = nn.Linear(config.d_model, config.n_classes)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0)
            torch.nn.init.constant_(module.weight, 1.0)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape

        x = self.input_proj(x)  # (batch, seq_len, d_model)

        x, _ = self.gru(x)  # (batch, seq_len, d_model)

        x = self.norm(x[:, -1, :])  # (batch, d_model)
        x = self.dropout(x)

        logits = self.classifier(x)  # (batch, n_classes)

        return logits


class FinancialDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.LongTensor([self.labels[idx]])


class DataPreprocessor:
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.label_mapping = {'up': 0, 'flat': 1, 'down': 2}

    def load_and_preprocess(self, file_path):
        df = pd.read_csv(file_path, parse_dates=['time'])
        df = df.sort_values('time').reset_index(drop=True)

        for i in range(1, 4):
            df[f'label-{i}'] = df['label'].shift(i)

        missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        df['label_encoded'] = df[TARGET_COLUMN].map(self.label_mapping)

        for lag_col in ['label-1', 'label-2', 'label-3']:
            df[lag_col] = df[lag_col].map(self.label_mapping)

        original_size = len(df)
        df = df.dropna(subset=FEATURE_COLUMNS + ['label_encoded'])
        new_size = len(df)
        # print(f"Deleted lines with NaN: {original_size - new_size} из {original_size}")

        feature_data = df[FEATURE_COLUMNS].values
        feature_data_scaled = self.scaler.fit_transform(feature_data)
        sequences, labels = self.create_sequences(feature_data_scaled, df['label_encoded'].values)

        return sequences, labels, df

    def create_sequences(self, data, labels):
        sequences = []
        target_labels = []

        for i in range(self.sequence_length, len(data)):
            seq = data[i - self.sequence_length:i]
            label = labels[i - 1]

            sequences.append(seq)
            target_labels.append(label)

        return np.array(sequences), np.array(target_labels)

    def get_class_weights(self, labels):
        class_counts = np.bincount(labels)
        total_samples = len(labels)
        weights = total_samples / (len(class_counts) * class_counts)
        return torch.tensor(weights, dtype=torch.float32)


def predict_new_data(model_path, new_data_path, sequence_length=60):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = FinancialSSM(config).to(device)

    # ВАЖНО: map_location=device, чтобы перегрузить CUDA-чекпоинт на CPU
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    preprocessor = DataPreprocessor(sequence_length=sequence_length)
    sequences, labels, df = preprocessor.load_and_preprocess(new_data_path)

    predictions = []
    confidence_scores = []
    all_probs = []

    test_dataset = FinancialDataset(sequences, labels)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            outputs = model(data)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            predictions.extend(preds.cpu().numpy())
            confidence_scores.extend(torch.max(probs, dim=1)[0].cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    label_mapping = {0: 'up', 1: 'flat', 2: 'down'}
    predicted_labels = [label_mapping[pred] for pred in predictions]

    results_df = df.iloc[sequence_length:].copy()
    results_df['predicted_label'] = predicted_labels
    results_df['confidence'] = confidence_scores
    results_df['prob_up'] = [p[0] for p in all_probs]
    results_df['prob_flat'] = [p[1] for p in all_probs]
    results_df['prob_down'] = [p[2] for p in all_probs]

    return results_df
