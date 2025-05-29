import neuralNetwork as debilNetwork
import DataLoader as dl
import torch
import matplotlib.pyplot as plt

df_testing = dl.load_all_data(dl.DYNAMIC)

df_testing = df_testing.dropna(
    subset=[
        "data__coordinates__x",
        "reference__x",
        "data__coordinates__y",
        "reference__y",
    ]
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

# 1. Recreate the model architecture exactly as before
model = debilNetwork.CorrectionNet()

# 2. Load the saved parameters
model.load_state_dict(torch.load("correction_model.pth"))

# 3. Set model to evaluation mode
model.eval()

# 4. Use model as usual
with torch.no_grad():
    corrected_test = model(inputs_test)

# Now you can calculate errors and plot as before
errors = corrected_test - targets_test

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(errors[:, 0].numpy(), bins=50, color="blue", alpha=0.7)
plt.xlabel("Error in X")
plt.ylabel("Frequency")
plt.title("Histogram of Correction Errors (X)")

plt.subplot(1, 2, 2)
plt.hist(errors[:, 1].numpy(), bins=50, color="blue", alpha=0.7)
plt.xlabel("Error in Y")
plt.ylabel("Frequency")
plt.title("Histogram of Correction Errors (Y)")

plt.tight_layout()
plt.show()
