import torch
import torch.nn as nn

# Use CrossEntropyLoss for binary classification
criterion = nn.CrossEntropyLoss()

# Optional: Use class weights if benign/malignant imbalance exists
def get_weighted_loss(num_benign, num_malignant):
    total = num_benign + num_malignant
    weight_benign = total / (2 * num_benign)
    weight_malignant = total / (2 * num_malignant)
    weights = torch.tensor([weight_benign, weight_malignant], dtype=torch.float32)
    return nn.CrossEntropyLoss(weight=weights)
