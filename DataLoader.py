import glob as glob
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def load_static_data():
    f8_path = "./pomiary/F8/"
    f10_path = "./pomiary/F10/"
    all_data = []

    # Collect stat files from both directories
    static_files = glob.glob(f"{f8_path}*stat*.xlsx") + glob.glob(
        f"{f10_path}*stat*.xlsx"
    )
    print(f"Znaleziono {len(static_files)} plików statycznych")

    for file_path in static_files:
        try:
            df = pd.read_excel(file_path, sheet_name="Sheet1")
            all_data.append(df)
        except Exception as e:
            print(f"Błąd ładowania {file_path}: {e}")

    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"Łączenie próbek statycznych: {len(combined_data)}")
        return combined_data

    print("Nie znaleziono danych statycznych.")
    return None


def load_dynamic_data():
    f8_path = "./pomiary/F8/"
    f10_path = "./pomiary/F10/"
    all_data = []

    # Get dynamic files from both folders
    dynamic_files = [
        f
        for f in glob.glob(f"{f8_path}*.xlsx") + glob.glob(f"{f10_path}*.xlsx")
        if "stat" not in f.lower() and "random" not in f.lower()
    ]
    print(f"Znaleziono {len(dynamic_files)} plików dynamicznych")

    for file_path in dynamic_files:
        try:
            df = pd.read_excel(file_path, sheet_name="Sheet1")
            all_data.append(df)
        except Exception as e:
            print(f"Błąd ładowania {file_path}: {e}")

    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"Łączenie próbek dynamicznych: {len(combined_data)}")
        return combined_data

    print("Nie znaleziono danych dynamicznych.")
    return None

    def prepare_data_excel(df):
        """Przygotowanie danych z formatu Excel"""

        X = df["data__coordinates__x"].values
        X = np.nan_to_num(X, nan=0.0)
        df["error_x"] = df["data__coordinates__x"] - df["reference__x"]
        df["error_y"] = df["data__coordinates__y"] - df["reference__y"]
        Y = df[["error_x", "error_y"]].values

        return X, Y


df = load_static_data()

df = df.dropna(subset=["data__coordinates__x", "reference__x"])
# Convert to PyTorch tensors
x_measured = torch.tensor(
    df["data__coordinates__x"].values, dtype=torch.float32
).unsqueeze(1)
x_reference = torch.tensor(df["reference__x"].values, dtype=torch.float32).unsqueeze(1)

print(x_measured)
print(x_reference)


# Define model
class CorrectionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


model = CorrectionNet()

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
n_epochs = 500
for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()

    output = model(x_measured)
    loss = criterion(output, x_reference)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Evaluate the correction
model.eval()
with torch.no_grad():
    corrected = model(x_measured)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(x_reference.numpy(), label="Reference (True)", color="green")
plt.plot(x_measured.numpy(), label="Measured (Noisy)", color="red", alpha=0.6)
plt.plot(corrected.numpy(), label="Corrected (NN Output)", color="blue")
plt.legend()
plt.title("Neural Network Correction")
plt.xlabel("Sample Index")
plt.ylabel("Value")
plt.grid(True)
plt.show()
