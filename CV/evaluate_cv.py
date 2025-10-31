"""
Evaluate the saved model on validation split taken from the same manifest logic
used by `CV/stock_vision_model.py`.

Outputs:
 - cv_predictions_val.csv (filename,true_label,pred_label,prob)
 - cv_confusion_matrix.png
 - cv_confusion_matrix.csv
 - cv_classification_report.txt

The script will try to use scikit-learn for metrics; if it's not installed it will
fall back to a small internal implementation.

Usage:
  python CV/evaluate_cv.py --manifest data/cv/images/SBER/5m/batch_0.csv --model best_stock_vision.pt

"""
import argparse
import os
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Import utilities and classes from the training script
try:
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from CV.stock_vision_model import load_data, StockChartDataset, StockVisionModel
except Exception as e:
    # try relative import
    try:
        from stock_vision_model import load_data, StockChartDataset, StockVisionModel
    except Exception as e2:
        raise RuntimeError("Couldn't import from stock_vision_model.py: {}".format(e2))


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def evaluate(manifest_path, model_path, batch_size=64, device='cpu'):
    print('Loading manifest and splitting (same logic as training) ...')
    _, val_df = load_data(manifest_path, val_ratio=0.2)
    print(f'Validation rows: {len(val_df)}')

    val_ds = StockChartDataset(val_df)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # Build model and load weights
    model = StockVisionModel(num_classes=3)
    map_loc = torch.device(device)
    state = torch.load(model_path, map_location=map_loc)
    try:
        model.load_state_dict(state)
    except Exception:
        # saved might be a dict with key 'model_state_dict' or similar
        if isinstance(state, dict):
            if 'model_state_dict' in state:
                model.load_state_dict(state['model_state_dict'])
            else:
                # try to load directly if shapes match
                model.load_state_dict(state)
        else:
            raise

    model.to(device)
    model.eval()

    y_true = []
    y_pred = []
    y_prob = []
    filenames = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_prob.extend(probs.max(dim=1).values.cpu().numpy().tolist())

    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    y_prob = np.array(y_prob, dtype=float)

    # Save predictions
    out_csv = 'cv_predictions_val.csv'
    print('Writing', out_csv)
    idxs = val_df.index.tolist()
    # attempt to write filenames from manifest rows
    rows = val_df.reset_index()
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['idx','path','true_label','pred_label','pred_prob'])
        for i, (t, p, pr) in enumerate(zip(y_true, y_pred, y_prob)):
            path = rows.loc[i, 'path'] if 'path' in rows.columns else ''
            w.writerow([rows.loc[i, 'index'] if 'index' in rows.columns else idxs[i], path, int(t), int(p), float(pr)])

    # confusion matrix & classification report
    num_classes = 3
    # try sklearn
    try:
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        report = classification_report(y_true, y_pred, labels=list(range(num_classes)))
        print('\nClassification report:\n')
        print(report)
    except Exception as e:
        print('scikit-learn not available or failed to import; falling back to simple metrics. (', e, ')')
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for a, b in zip(y_true, y_pred):
            cm[int(a), int(b)] += 1

        # simple precision/recall/f1
        prec = []
        rec = []
        f1 = []
        for cls in range(num_classes):
            tp = cm[cls, cls]
            fp = cm[:, cls].sum() - tp
            fn = cm[cls, :].sum() - tp
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            _f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            prec.append(p); rec.append(r); f1.append(_f1)
        report_lines = [f'Class {i}: precision={prec[i]:.3f}, recall={rec[i]:.3f}, f1={f1[i]:.3f}' for i in range(num_classes)]
        report = '\n'.join(report_lines)
        print('\nFallback classification report:\n')
        print(report)

    # Save confusion matrix numeric CSV
    cm_csv = 'cv_confusion_matrix.csv'
    print('Writing', cm_csv)
    np.savetxt(cm_csv, cm, fmt='%d', delimiter=',')

    # Plot confusion matrix
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix (val)')
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, [str(i) for i in range(num_classes)])
    plt.yticks(tick_marks, [str(i) for i in range(num_classes)])

    thresh = cm.max() / 2. if cm.max()>0 else 1
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment='center',
                     color='white' if cm[i, j] > thresh else 'black')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    png_out = 'cv_confusion_matrix.png'
    plt.savefig(png_out)
    print('Wrote', png_out)

    # Save classification report text
    rep_out = 'cv_classification_report.txt'
    with open(rep_out, 'w') as f:
        f.write(report)
    print('Wrote', rep_out)

    # Print summary accuracy
    acc = (y_true == y_pred).mean()
    print('\nOverall val accuracy: {:.4f}'.format(acc))
    print('Done.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=str, default='data/cv/images/SBER/5m/batch_0.csv')
    p.add_argument('--model', type=str, default='best_stock_vision.pt')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--device', type=str, default='cpu')
    args = p.parse_args()

    if not os.path.exists(args.manifest):
        raise SystemExit(f"Manifest not found: {args.manifest}")
    if not os.path.exists(args.model):
        print(f"Model file '{args.model}' not found in current directory. You can pass --model with path to your saved weights.")

    evaluate(args.manifest, args.model, batch_size=args.batch_size, device=args.device)
