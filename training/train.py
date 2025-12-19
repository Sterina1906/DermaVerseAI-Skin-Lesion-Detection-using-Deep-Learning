import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import cv2
import os
from tqdm import tqdm
from datetime import datetime


from models.binary_classifier import BinaryClassifier
from training.augmentation import get_train_transforms, get_val_transforms
from training.losses import get_weighted_loss
from training.metrics import calculate_metrics, get_predictions
UNFREEZE_EPOCH = 4  # unfreeze after epoch 4

# ========== CUSTOM DATASET ==========
class SkinLesionDataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['filename']
        label = row['label']
        
        # Load image (adjust path logic based on your structure)
        img_path = os.path.join(self.data_dir, filename)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, torch.tensor(label, dtype=torch.long)

# ========== TRAINING LOOP ==========
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = get_predictions(outputs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    metrics = calculate_metrics(all_preds, all_labels)
    
    return avg_loss, metrics

# ========== VALIDATION LOOP ==========
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = get_predictions(outputs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    metrics = calculate_metrics(all_preds, all_labels)
    
    return avg_loss, metrics

# ========== MAIN TRAINING ==========
if __name__ == "__main__":
    # Hyperparameters
    EPOCHS = 20
    BATCH_SIZE = 16  # Adjust for 4GB GPU
    LEARNING_RATE = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {DEVICE}")
    
    # Create datasets and dataloaders
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()
    
    train_dataset = SkinLesionDataset('data/splits/train.csv', 'data/raw/images/', train_transform)
    val_dataset = SkinLesionDataset('data/splits/val.csv', 'data/raw/images/', val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Create model
    model = BinaryClassifier(pretrained=True, freeze_backbone=True)
    model.to(DEVICE)
    
    # Loss and optimizer
    train_df = pd.read_csv("data/splits/train.csv")
    num_benign = (train_df["label"] == 0).sum()
    num_malignant = (train_df["label"] == 1).sum()

    criterion = get_weighted_loss(num_benign, num_malignant).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    for epoch in range(EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Train Acc: {train_metrics['accuracy']:.4f} | F1: {train_metrics['f1']:.4f}")
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, DEVICE)
        print(f"\nVal Loss: {val_loss:.4f}")
        print(f"Val Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f}")

        # 🔓 Unfreeze backbone after a few epochs
        if epoch + 1 == UNFREEZE_EPOCH:
            print("\n🔓 Unfreezing backbone and lowering learning rate")
            model.unfreeze_backbone()
            for g in optimizer.param_groups:
                g["lr"] = 1e-5
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f'models/weights/best_model.pth')
            print("✅ Model saved!")
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print("⏹️  Early stopping triggered!")
                break
    
    print("\n✅ Training complete!")
    print(f"Best model saved to: models/weights/best_model.pth")
