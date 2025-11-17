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
    data_path = "SBER_dataset_5m.csv"
    sequence_length = 60
    input_dim = 12
    d_model = 256
    d_state = 16
    n_layers = 4
    n_classes = 3
    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 50
    dropout_rate = 0.1
    weight_decay = 1e-5
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    
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
        print(f"Deleted lines with NaN: {original_size - new_size} из {original_size}")
        
        feature_data = df[FEATURE_COLUMNS].values
        feature_data_scaled = self.scaler.fit_transform(feature_data)
        sequences, labels = self.create_sequences(feature_data_scaled, df['label_encoded'].values)
        
        return sequences, labels, df
    
    def create_sequences(self, data, labels):
        sequences = []
        target_labels = []
        
        for i in range(self.sequence_length, len(data)):
            seq = data[i-self.sequence_length:i]
            label = labels[i-1]
            
            sequences.append(seq)
            target_labels.append(label)
            
        return np.array(sequences), np.array(target_labels)
    
    def get_class_weights(self, labels):
        class_counts = np.bincount(labels)
        total_samples = len(labels)
        weights = total_samples / (len(class_counts) * class_counts)
        return torch.tensor(weights, dtype=torch.float32)

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, class_weights):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.class_weights = class_weights
        
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(device), target.squeeze().to(device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(device), target.squeeze().to(device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                preds = torch.argmax(output, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        accuracy = accuracy_score(all_targets, all_preds)
        return total_loss / len(self.val_loader), accuracy, all_preds, all_targets
    
    def train(self):
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        for epoch in range(self.config.num_epochs):
            train_loss = self.train_epoch()
            val_loss, val_accuracy, _, _ = self.validate()
            
            self.scheduler.step(val_loss)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            
            print(f'Epoch {epoch+1}/{config.num_epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_ssm_model.pth')
                print(f'  -> New best model with val_loss: {val_loss:.4f}')
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping on epoch {epoch+1}")
                    break
    
    def plot_training_history(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_title('Training history')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        ax2.plot(self.val_accuracies, label='Val Accuracy', color='green')
        ax2.set_title('Accuracy on validation')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Load data...")
print(f"Using features: {FEATURE_COLUMNS}")

preprocessor = DataPreprocessor(sequence_length=config.sequence_length)
sequences, labels, df = preprocessor.load_and_preprocess(config.data_path)

print(f"Data shape: {sequences.shape}")
print(f"Class distribution: {np.bincount(labels)}")

total_samples = len(sequences)
train_size = int(config.train_ratio * total_samples)
val_size = int(config.val_ratio * total_samples)
test_size = total_samples - train_size - val_size

indices = np.arange(total_samples)
train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]

train_sequences = sequences[train_indices]
train_labels = labels[train_indices]
val_sequences = sequences[val_indices]
val_labels = labels[val_indices]
test_sequences = sequences[test_indices]
test_labels = labels[test_indices]

print(f"Sample size - Train: {len(train_sequences)}, Val: {len(val_sequences)}, Test: {len(test_sequences)}")

train_dataset = FinancialDataset(train_sequences, train_labels)
val_dataset = FinancialDataset(val_sequences, val_labels)
test_dataset = FinancialDataset(test_sequences, test_labels)
    
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

class_weights = preprocessor.get_class_weights(labels)
print(f"Class weights: {class_weights}")

model = FinancialSSM(config).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total_params:,} (trainable: {trainable_params:,})")

trainer = Trainer(model, train_loader, val_loader, config, class_weights.to(device))
trainer.train()

trainer.plot_training_history()

model.load_state_dict(torch.load('best_ssm_model.pth'))
model.eval()

test_preds = []
test_targets = []
test_probs = []

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.squeeze().to(device)
        output = model(data)
        probs = torch.softmax(output, dim=1)
        preds = torch.argmax(output, dim=1)
        test_preds.extend(preds.cpu().numpy())
        test_targets.extend(target.cpu().numpy())
        test_probs.extend(probs.cpu().numpy())

test_accuracy = accuracy_score(test_targets, test_preds)
print(f"Accuracy on test: {test_accuracy:.4f}")

target_names = ['up', 'flat', 'down']
print("\nClassification Report:")
print(classification_report(test_targets, test_preds, target_names=target_names))


def predict_new_data(model_path, new_data_path, sequence_length=60):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = FinancialSSM(config).to(device)
    model.load_state_dict(torch.load(model_path))
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
