import DataLoader as dl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# --- Load and clean data ---
df_training, df_testing = dl.load_all_data()

df_training = df_training.dropna(
    subset=[
        "data__coordinates__x",
        "reference__x",
        "data__coordinates__y",
        "reference__y",
    ]
)
df_testing = df_testing.dropna(
    subset=[
        "data__coordinates__x",
        "reference__x",
        "data__coordinates__y",
        "reference__y",
    ]
)

# --- Prepare training data ---
inputs_train = torch.tensor(
    df_training[["data__coordinates__x", "data__coordinates__y"]].values,
    dtype=torch.float32,
)
targets_train = torch.tensor(
    df_training[["reference__x", "reference__y"]].values,
    dtype=torch.float32,
)

# --- Prepare test data ---
inputs_test = torch.tensor(
    df_testing[["data__coordinates__x", "data__coordinates__y"]].values,
    dtype=torch.float32,
)
targets_test = torch.tensor(
    df_testing[["reference__x", "reference__y"]].values,
    dtype=torch.float32,
)


# --- Define model ---
class CorrectionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, xy):
        return self.net(xy)


model = CorrectionNet()

# --- Loss and optimizer ---
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Training loop ---
n_epochs = 1000
for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()

    output = model(inputs_train)
    loss = criterion(output, targets_train)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# --- Evaluate on test set ---
model.eval()
with torch.no_grad():
    corrected_test = model(inputs_test)

# --- Plot results: x and y scatter plots ---
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

axs[0].scatter(
    range(len(targets_test)),
    targets_test[:, 0].numpy(),
    label="Reference X",
    color="green",
    s=10,
)
axs[0].scatter(
    range(len(inputs_test)),
    inputs_test[:, 0].numpy(),
    label="Measured X",
    color="red",
    alpha=0.5,
    s=10,
)
axs[0].scatter(
    range(len(corrected_test)),
    corrected_test[:, 0].numpy(),
    label="Corrected X",
    color="blue",
    s=10,
)
axs[0].set_title("X Axis Correction")
axs[0].set_xlabel("Sample Index")
axs[0].set_ylabel("X Value")
axs[0].grid(True)
axs[0].legend()

axs[1].scatter(
    range(len(targets_test)),
    targets_test[:, 1].numpy(),
    label="Reference Y",
    color="green",
    s=10,
)
axs[1].scatter(
    range(len(inputs_test)),
    inputs_test[:, 1].numpy(),
    label="Measured Y",
    color="red",
    alpha=0.5,
    s=10,
)
axs[1].scatter(
    range(len(corrected_test)),
    corrected_test[:, 1].numpy(),
    label="Corrected Y",
    color="blue",
    s=10,
)
axs[1].set_title("Y Axis Correction")
axs[1].set_xlabel("Sample Index")
axs[1].set_ylabel("Y Value")
axs[1].grid(True)
axs[1].legend()

plt.suptitle("Neural Network Correction: 2D Coordinates")
plt.tight_layout()
plt.show()
