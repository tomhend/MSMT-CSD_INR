from typing import Callable
from functools import partial
import torch.nn


def neg_con_mse_loss(
    output: torch.Tensor,
    labels: torch.Tensor,
    neg_values: torch.Tensor,
    _lambda: torch.Tensor,
    *args,
    **kwargs
) -> torch.Tensor:
    loss = ((output - labels) ** 2).sum(dim=1).mean()
    loss += (neg_values**2).sum(dim=1).mean() * 1
    return loss


def neg_con_mse_loss_ms(
    output: torch.Tensor,
    labels: torch.Tensor,
    neg_values: torch.Tensor,
    _lambda: torch.Tensor,
    *args,
    **kwargs
) -> torch.Tensor:
    n_grads = (labels.shape[-1] - 1)/2 # rough estimate of the number of directions per shell (unreliable)
    loss = (output - labels) ** 2
    loss[:, -1] = loss[:, -1]*n_grads # weigh b0's
    loss = loss.sum(dim=1).mean()
    loss += (neg_values**2).sum(dim=1).mean() * 1
    return loss


def create_neg_con_mse_loss(cfg: dict, *args, **kwargs) -> Callable:
    _lambda = torch.tensor(cfg["train_cfg"]["lambda"], dtype=torch.float)
    return partial(neg_con_mse_loss, _lambda=_lambda)


def create_neg_con_mse_loss_ms(cfg: dict, *args, **kwargs) -> Callable:
    _lambda = torch.tensor(cfg["train_cfg"]["lambda"], dtype=torch.float)
    return partial(neg_con_mse_loss_ms, _lambda=_lambda)


LOSS_FUNCTIONS = {
    "singleshell_csd_loss": create_neg_con_mse_loss,
    "multishell_csd_loss": create_neg_con_mse_loss_ms,
}


def get_loss_function(cfg: dict, *args, **kwargs) -> Callable:
    constructor = LOSS_FUNCTIONS.get(cfg["loss_function_name"], None)
    if constructor is None:
        raise Exception("Loss function name not recognized")
    return constructor(cfg, args, kwargs)
