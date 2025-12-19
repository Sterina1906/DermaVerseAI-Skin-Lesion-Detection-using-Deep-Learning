import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms():
    """Augmentation for training data"""
    return A.Compose([
        A.Resize(224, 224),
        A.Rotate(limit=20, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.GaussNoise(p=0.2),
        A.Blur(blur_limit=3, p=0.1),
        A.Normalize(mean=[0.485, 0.456, 0.406], 
                   std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=None)

def get_val_transforms():
    """No augmentation for validation/test"""
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], 
                   std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=None)
