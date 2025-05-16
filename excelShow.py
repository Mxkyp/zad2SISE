import pandas as pd
import torch
from torch.utils import data as data_utils
from sklearn.preprocessing import StandardScaler

"""
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
"""

# Load Excel file and sheet
xls = pd.ExcelFile("./pomiary/F8/f8_stat_10.xlsx")
xls2 = pd.ExcelFile("./pomiary/F8/f8_stat_11.xlsx")
df = xls.parse("Sheet1")

# === INPUT FEATURES ===
feature_cols = [
    "data__tagData__gyro__x",
    "data__tagData__gyro__y",
    "data__tagData__gyro__z",
    "data__tagData__magnetic__x",
    "data__tagData__magnetic__y",
    "data__tagData__magnetic__z",
    "data__tagData__quaternion__x",
    "data__tagData__quaternion__y",
    "data__tagData__quaternion__z",
    "data__tagData__quaternion__w",
    "data__tagData__linearAcceleration__x",
    "data__tagData__linearAcceleration__y",
    "data__tagData__linearAcceleration__z",
    "data__tagData__pressure",
]
X = df[feature_cols].values

# Normalize inputs
scaler = StandardScaler()
X = scaler.fit_transform(X)

# === TARGET ERRORS ===
df["error_x"] = df["data__coordinates__x"] - df["reference__x"]
df["error_y"] = df["data__coordinates__y"] - df["reference__y"]
Y = df[["error_x", "error_y"]].values

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

# Dataset & DataLoader
train = data_utils.TensorDataset(X_tensor, Y_tensor)
train_loader = data_utils.DataLoader(train, batch_size=50, shuffle=True)
