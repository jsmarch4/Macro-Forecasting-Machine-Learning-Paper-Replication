import numpy as np
import torch


def pinball_loss_torch(y_true, y_pred, tau):
    error = y_true - y_pred
    return torch.sum(torch.maximum(tau * error, (tau - 1) * error))


def pinball_loss_numpy(y_true, y_pred, tau):
    error = y_true - y_pred
    return np.where(error >= 0, tau * error, (tau - 1) * error)