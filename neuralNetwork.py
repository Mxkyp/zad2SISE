import torch
import torch.nn as nn
import torch.optim as optim


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


def train_model(inputs_train, targets_train, n_epochs=1000):
    model = CorrectionNet()

    # --- Loss and optimizer ---
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- Training loop ---
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        output = model(inputs_train)
        loss = criterion(output, targets_train)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # save model
    torch.save(model.state_dict(), "correction_model.pth")
