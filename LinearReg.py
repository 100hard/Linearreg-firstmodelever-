# linear_regression.py
import torch
from torch import nn
import matplotlib.pyplot as plt

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Create data
weight = 0.7
bias = 0.3
start = 0
end = 1
step = 0.02
x = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * x + bias

# Split data
train_split = int(0.8 * len(x))
x_train, y_train = x[:train_split], y[:train_split]
x_test, y_test = x[train_split:], y[train_split:]

# Plotting function
def plot_predictions(train_data=x_train,
                     train_labels=y_train,
                     test_data=x_test,
                     test_labels=y_test,
                     predictions=None):
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
    plt.legend(prop={"size": 14})
    plt.show()

# Build model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

# Initialize model
torch.manual_seed(67)
model = LinearRegressionModel()
model.to(device)

# Loss function and optimizer
loss_func = nn.L1Loss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)

# Training loop
torch.manual_seed(57)
epochs = 200

for epoch in range(epochs):
    model.train()
    y_pred = model(x_train.to(device))
    loss = loss_func(y_pred, y_train.to(device))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Testing
    model.eval()
    with torch.inference_mode():
        test_pred = model(x_test.to(device))
        test_loss = loss_func(test_pred, y_test.to(device))
    
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss.item()} | Test Loss: {test_loss.item()}")

# Final predictions
model.eval()
with torch.inference_mode():
    y_preds = model(x_test.to(device))

# Plot results
plot_predictions(predictions=y_preds.cpu())