import torch

from losses import pinball_loss_torch, pinball_loss_numpy


def train_model(model, X_train_tensor, y_train_tensor, tau, lam, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()

        y_pred = model(X_train_tensor)
        loss = pinball_loss_torch(y_train_tensor, y_pred, tau)

        beta = model.linear.weight.squeeze()
        ridge_penalty = lam * torch.sum(beta ** 2)

        total_loss = loss + ridge_penalty

        total_loss.backward()
        optimizer.step()

    return model


def average_pinball_loss(forecast_df, tau):
    forecast_col = f"q{tau:.2f}"

    losses = pinball_loss_numpy(
        forecast_df["actual"].to_numpy(),
        forecast_df[forecast_col].to_numpy(),
        tau
    )

    return losses.mean()