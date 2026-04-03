import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Parâmetros
# ---------------------------------------------------------------------------

mlp_params = {
    "input_dim": 4,
    "hidden_dims": [64, 32],
    "dropout": 0.3,
    "lr": 1e-3,
    "epochs": 50,
    "batch_size": 256,
    "random_state": 42,
}

# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------


class MLPClassifier(nn.Module):
    """
    Multi-Layer Perceptron for binary classification.

    Builds a sequential network with ReLU activations and Dropout
    between hidden layers, followed by a single logit output.

    Args:
        input_dim (int): Number of input features.
        hidden_dims (list[int]): Sizes of hidden layers.
        dropout (float): Dropout probability applied after each hidden layer.
    """

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def build_mlp() -> MLPClassifier:
    """
    Instantiate an MLPClassifier with the project's default params.

    Returns:
        MLPClassifier: Configured but untrained model.
    """
    return MLPClassifier(
        input_dim=mlp_params["input_dim"],
        hidden_dims=mlp_params["hidden_dims"],
        dropout=mlp_params["dropout"],
    )
