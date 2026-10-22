"""Training losses, written with the same ops as the TF originals."""
import torch


def nmse_loss(xhat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """GM-LAMP / LADMM scripts: mean_b sum_n |xhat-x|^2 / mean_b sum_n |x|^2."""
    num = torch.mean(torch.sum(torch.square(torch.abs(xhat - x)), 0))
    den = torch.mean(torch.sum(torch.square(torch.abs(x)), 0))
    return num / den


def l2_loss(xhat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """tf.nn.l2_loss(xhat - x) = sum(|xhat-x|^2)/2 (Borgerding / ALISTA scripts)."""
    d = xhat - x
    if d.is_complex():
        return torch.sum(torch.square(torch.abs(d))) / 2
    return torch.sum(d * d) / 2


def nmse_l2(xhat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Borgerding / ALISTA reporting NMSE: l2_loss(xhat-x)/l2_loss(x)."""
    return l2_loss(xhat, x) / l2_loss(torch.zeros_like(x), -x)


def log_nse_loss(xhat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Mean over samples of 10 log10(||xhat - x||^2 / ||x||^2) (dB; NF-LADMM proposal): every SNR weighs the same."""
    e = torch.sum(torch.square(torch.abs(xhat - x)), 0) / torch.sum(torch.square(torch.abs(x)), 0)
    return 10 * torch.mean(torch.log10(e + 1e-8))


def mse_loss(xhat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """E ||xhat - x||^2: mean over the columns of the squared error (UTRR, eq. 24)."""
    return torch.mean(torch.sum(torch.square(torch.abs(xhat - x)), 0))


LOSSES = {"nmse": nmse_loss, "l2": l2_loss, "log_nse": log_nse_loss, "mse": mse_loss}
