import argparse
import os
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class CVImageDataset(Dataset):
    def __init__(self, df, transform=None, to_rgb=True):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.to_rgb = to_rgb

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        p = self.df.loc[idx, 'path']
        img = Image.open(p)
        if self.to_rgb:
            img = img.convert('RGB')
        x = self.transform(img) if self.transform is not None else transforms.ToTensor()(img)
        y = int(self.df.loc[idx, 'label_idx'])
        return x, torch.tensor(y, dtype=torch.long)

def make_dataloaders(labels_csv='data/labels.csv', images_root='data/cv', batch_size=32, 
                    subset=None, val_frac=0.2, num_workers=4, seed=42):
    df = pd.read_csv(labels_csv)
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    label_map = {lab: i for i, lab in enumerate(sorted(df['label'].unique()))}
    df['label_idx'] = df['label'].map(label_map)
    df['path'] = df['filename'].apply(lambda x: os.path.join(images_root, x))

    if subset is not None and subset > 0:
        df = df.sample(n=min(subset, len(df)), random_state=seed).reset_index(drop=True)

    train_df, val_df = train_test_split(
        df, test_size=val_frac, stratify=df['label_idx'], 
        random_state=seed, shuffle=True
    )

    train_t = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=2),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_t = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = CVImageDataset(train_df, transform=train_t, to_rgb=True)
    val_ds = CVImageDataset(val_df, transform=val_t, to_rgb=True)

    counts = train_df['label_idx'].value_counts().sort_index().values
    counts = np.array([max(c, 1) for c in counts], dtype=float)
    class_weights = 1.0 / counts
    sample_weights = train_df['label_idx'].map(lambda x: class_weights[int(x)]).values
    sample_weights = torch.DoubleTensor(sample_weights)
    from torch.utils.data import WeightedRandomSampler
    sampler = WeightedRandomSampler(
        sample_weights, 
        num_samples=len(sample_weights), 
        replacement=True
    )

    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        sampler=sampler,
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )

    meta = {
        'label_map': label_map, 
        'train_len': len(train_ds), 
        'val_len': len(val_ds),
        'class_counts': dict(df['label_idx'].value_counts().sort_index())
    }
    return train_loader, val_loader, meta

def create_model(num_classes, device):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_feat = model.fc.in_features
    
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_feat, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
        nn.Linear(512, num_classes)
    )
    
    for m in model.fc.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    return model.to(device)

def train_epoch(model, train_loader, criterion, optimizer, device, accumulation_steps=4):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    
    for i, (xb, yb) in enumerate(train_loader):
        xb, yb = xb.to(device), yb.to(device)
        
        outputs = model(xb)
        loss = criterion(outputs, yb) / accumulation_steps
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        running_loss += loss.item() * xb.size(0) * accumulation_steps
        _, predicted = outputs.max(1)
        total += yb.size(0)
        correct += predicted.eq(yb).sum().item()
    
    if len(train_loader) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
    
    train_loss = running_loss / total
    train_acc = correct / total
    return train_loss, train_acc

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            loss = criterion(outputs, yb)
            
            val_loss += loss.item() * xb.size(0)
            _, predicted = outputs.max(1)
            total += yb.size(0)
            correct += predicted.eq(yb).sum().item()
    
    val_loss /= total
    val_acc = correct / total
    return val_loss, val_acc

def plot_training_history(train_losses, val_accs, train_accs):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Train')
    plt.plot(val_accs, label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(val_accs)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.axhline(y=1/3, color='r', linestyle='--', label='Random (33%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('models/CV/training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(model, val_loader, device, label_map):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    class_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('models/CV/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    return cm

def train(train_loader, val_loader, model, device, epochs=30, batch_size=32, label_map=None):
    save_path = 'models/CV/best_cv_model.pth'
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    
    class FocalLoss(nn.Module):
        def __init__(self, alpha=1, gamma=2, reduction='mean'):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.reduction = reduction
            
        def forward(self, inputs, targets):
            ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
            
            if self.reduction == 'mean':
                return focal_loss.mean()
            elif self.reduction == 'sum':
                return focal_loss.sum()
            else:
                return focal_loss
    
    criterion = FocalLoss(gamma=2.0)
    
    backbone_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'fc' in name:
            classifier_params.append(param)
        else:
            backbone_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': 1e-4},
        {'params': classifier_params, 'lr': 3e-4}
    ], weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 12
    
    train_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(epochs):
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        scheduler.step()
        
        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{epochs}  time={epoch_time:.1f}s  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
              f"val_acc={val_acc:.3f}  lr={current_lr:.2e}")
        
        if val_acc > best_val_acc + 0.001:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc
            }, save_path)
            patience_counter = 0
            print(f"✓ New best! Val accuracy: {val_acc:.4f}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{max_patience})")
        
        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    print(f'Training finished. Best validation accuracy: {best_val_acc:.4f}')
    
    plot_training_history(train_losses, val_accs, train_accs)
    
    print("\nGenerating confusion matrix...")
    model.load_state_dict(torch.load(save_path)['model_state_dict'])
    plot_confusion_matrix(model, val_loader, device, label_map)
    
    return best_val_acc

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    train_loader, val_loader, meta = make_dataloaders(
        batch_size=args.batch_size,
        num_workers=4
    )
    
    print('Label map:', meta['label_map'])
    print(f'Train samples: {meta["train_len"]}, Val samples: {meta["val_len"]}')
    
    model = create_model(num_classes=len(meta['label_map']), device=device)
    
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    best_acc = train(
        train_loader, 
        val_loader, 
        model, 
        device, 
        epochs=args.epochs,
        batch_size=args.batch_size,
        label_map=meta['label_map']
    )
    
    print(f'\nFinal best validation accuracy: {best_acc:.4f}')

if __name__ == '__main__':
    main()