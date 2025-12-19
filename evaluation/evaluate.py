import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
import matplotlib.pyplot as plt

from models.binary_classifier import BinaryClassifier
from training.augmentation import get_val_transforms
from training.train import SkinLesionDataset
from training.metrics import get_predictions

def evaluate_model(model_path, test_csv, data_dir, device):
    """
    Evaluate model on test set
    """
    # Load model
    model = BinaryClassifier(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Create test dataset
    test_transform = get_val_transforms()
    test_dataset = SkinLesionDataset(test_csv, data_dir, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Get predictions
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            preds = get_predictions(outputs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of malignant
    
    # Calculate metrics
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)
    
    print(f"\nAccuracy:  {accuracy_score(all_labels, all_preds):.4f}")
    print(f"Precision: {precision_score(all_labels, all_preds):.4f}")
    print(f"Recall:    {recall_score(all_labels, all_preds):.4f}")
    print(f"F1-Score:  {f1_score(all_labels, all_preds):.4f}")
    print(f"AUC-ROC:   {roc_auc_score(all_labels, all_probs):.4f}")
    
    print("\n" + "-"*60)
    print("CLASSIFICATION REPORT")
    print("-"*60)
    print(classification_report(all_labels, all_preds, 
          target_names=['Benign (0)', 'Malignant (1)']))
    
    print("\nCONFUSION MATRIX")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    print(f"\nTrue Negatives:  {cm}")
    print(f"False Positives: {cm}")
    print(f"False Negatives: {cm}")
    print(f"True Positives:  {cm}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate_model(
        model_path='models/weights/best_model.pth',
        test_csv='data/splits/test.csv',
        data_dir='data/raw/images/',
        device=device
    )
