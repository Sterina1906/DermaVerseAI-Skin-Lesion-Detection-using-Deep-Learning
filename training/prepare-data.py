import pandas as pd
from sklearn.model_selection import train_test_split
import os

# 1. Read the CSV
df = pd.read_csv('data/raw/ISIC2018_Task3_Training_GroundTruth.csv')


# 2. Map 7 classes to Binary (Benign=0, Malignant=1)
# The CSV has columns: image, MEL, NV, BCC, AKIEC, BKL, DF, VASC
def get_binary_label(row):
    # Malignant classes: MEL, BCC, AKIEC
    if row['MEL'] == 1 or row['BCC'] == 1 or row['AKIEC'] == 1:
        return 1
    # All others are Benign
    return 0

df['label'] = df.apply(get_binary_label, axis=1)

# 3. Add .jpg extension
df['filename'] = df['image'] + '.jpg'  # 'image' is the column name in ISIC 2018 CSV

# 4. Keep only what we need
df = df[['filename', 'label']]

print(f"\nTotal images: {len(df)}")
print(f"Benign: {(df['label']==0).sum()} | Malignant: {(df['label']==1).sum()}")

# Split into train (70%), val (15%), test (15%)
train, test = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
val, test = train_test_split(test, test_size=0.5, random_state=42, stratify=test['label'])

# Save splits
os.makedirs('data/splits', exist_ok=True)
train.to_csv('data/splits/train.csv', index=False)
val.to_csv('data/splits/val.csv', index=False)
test.to_csv('data/splits/test.csv', index=False)

print(f"\n✅ Data split complete!")
print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")