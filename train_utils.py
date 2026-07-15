import torch

from losses import pinball_loss_torch, pinball_loss_numpy


def train_model(model, X_train_tensor, y_train_tensor, tau, lam, epochs, lr):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()

        y_pred = model(X_train_tensor)

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

        # torch.nn.utils.clip_grad_norm_(
        #     model.parameters(),
        #     max_norm=1.0
        # )

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