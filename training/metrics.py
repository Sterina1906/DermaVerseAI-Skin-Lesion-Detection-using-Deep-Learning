import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def calculate_metrics(preds, labels):
    """
    Calculate: accuracy, precision, recall, F1, AUC
    preds: predicted class (0 or 1)
    labels: true class (0 or 1)
    """
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
    }

def get_predictions(outputs):
    """Convert model output logits to class predictions"""
    return torch.argmax(outputs, dim=1).cpu().numpy()
