import torch
import torch.nn as nn
import torchvision as tv
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class ChartDataset(Dataset):
    def __init__(self, frame):
        self.df = frame.reset_index(drop=True)
        self.transform = tv.transforms.Compose([
            tv.transforms.Resize((224, 224)),
            tv.transforms.Grayscale(num_output_channels=1),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([0.5], [0.5])
        ])
            
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, i):
        img = Image.open(self.df.path[i]).convert('L')
        x = self.transform(img)
        y = int(self.df.label[i])
        return x, y

class CompVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.4),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 3)
        )
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def train_model(train_loader, val_loader, epochs=30, device='cuda'):
    model = CompVisionModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = total = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, pred = out.max(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                _, pred = out.max(1)
                val_correct += (pred == y).sum().item()
                val_total += y.size(0)
                
        train_acc = correct / total
        val_acc = val_correct / val_total
        avg_loss = train_loss/len(train_loader)
        print(f'Epoch {epoch}: loss={avg_loss:.4f}, train_acc={train_acc:.3f}, val_acc={val_acc:.3f}')
        
    torch.save(model.state_dict(), 'stock_vision.pt')
    return model

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    df = pd.read_csv('data/labels.csv')
    
    # Convert text labels to numeric
    label_map = {'down': 0, 'flat': 1, 'up': 2}
    df['label'] = df['label'].map(label_map)
    
    # Add the full path to images
    df['path'] = df['filename'].apply(lambda x: f'data/cv/{x}')
    
    msk = np.random.rand(len(df)) < 0.8
    train_df = df[msk].reset_index(drop=True)
    val_df = df[~msk].reset_index(drop=True)

    train_ds = ChartDataset(train_df)
    val_ds = ChartDataset(val_df)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = train_model(train_loader, val_loader, device=device)

if __name__ == '__main__':
    main()