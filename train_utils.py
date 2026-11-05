
import torch

from losses import pinball_loss_torch, pinball_loss_numpy


def train_model(model, X_train_tensor, y_train_tensor,
                tau, lam, epochs, lr, optimizer=None):

    # Create Adam only if an optimizer was not provided
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()

        y_pred = model(X_train_tensor).reshape(-1)

        # to make sure shapes of y_pred and y_true are the same stop otherwise
        assert y_pred.shape == y_train_tensor.shape, (
        f"Shape mismatch: y_pred={y_pred.shape}, "
        f"y_true={y_train_tensor.shape}"
)

        pinball = pinball_loss_torch(
            y_train_tensor,
            y_pred,
            tau
        )

        if hasattr(model, "l2_penalty"):
            l2 = model.l2_penalty()
        else:
            l2 = torch.sum(model.linear.weight.squeeze() ** 2)

        total_loss = pinball + lam * l2

        total_loss.backward()
        optimizer.step()

    return model, optimizer


def average_pinball_loss(forecast_df, tau):
    forecast_col = f"q{tau:.2f}"
    losses = pinball_loss_numpy(
        forecast_df["actual"].to_numpy(),
        forecast_df[forecast_col].to_numpy(),
        tau,
    )

    return float(losses.mean())
