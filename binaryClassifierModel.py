import torch
import torch.nn as nn

class PCABinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # Update in_features to 6 since we now have 6 features (2 original + 4 intricate)
        self.layer_1 = nn.Linear(in_features=6, out_features=8)
        self.layer_2 = nn.Linear(in_features=8, out_features=1)  # Assuming binary classification
        self.relu = nn.ReLU()

    def forward(self, x):
        # Calculate intricate features inside the model
        x1 = x[:, 0]  # PCA1
        x2 = x[:, 1]  # PCA2

        # Create intricate features
        x1_squared = x1 ** 2
        x2_squared = x2 ** 2
        sin_x1 = torch.sin(x1)
        sin_x2 = torch.sin(x2)

        # Concatenate original features and intricate features
        x = torch.stack((x1, x2, x1_squared, x2_squared, sin_x1, sin_x2), dim=1)

        # Forward pass through the layers
        x = self.relu(self.layer_1(x))
        return self.layer_2(x)
