import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class BinaryClassifier(nn.Module):
    """
    EfficientNet-B0 fine-tuned for binary classification (benign vs malignant)
    """
    def __init__(self, pretrained=True, freeze_backbone=False):
        super().__init__()

        # Load pretrained EfficientNet-B0
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)

        # Original head is something like:
        # self.backbone.classifier = Sequential(
        #     Dropout(p=0.2, inplace=True),
        #     Linear(in_features=1280, out_features=1000, bias=True)
        # )

        # ✅ Get in_features from the Linear layer (index 1)
        in_features = self.backbone.classifier[1].in_features

        # ✅ Replace with 2‑class head
        self.backbone.classifier[1] = nn.Linear(in_features, 2)

        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.backbone(x)

    def unfreeze_backbone(self):
        for param in self.backbone.features.parameters():
            param.requires_grad = True
