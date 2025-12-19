import pandas as pd

df = pd.read_csv("data/splits/train.csv")
num_benign = (df["label"] == 0).sum()
num_malignant = (df["label"] == 1).sum()
print("Benign:", num_benign, "Malignant:", num_malignant)
