import neuralNetwork as debilNetwork
import DataLoader as dl
import plotter as pl
import torch

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

pl.plot_error_distributions(targets_test, corrected_test, "test")
