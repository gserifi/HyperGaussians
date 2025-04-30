from typing import Tuple, Union

import torch
import torch.nn as nn

HGSOutput = Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


class HyperGaussians(nn.Module):
    """Computes the conditional means and uncertainties of a set of Gaussians given the input.

    This module internally represents the conditional Gaussians using their mean μ = (μ_o, μ_i) and the relevant parts
    of the Cholesky factors of their precision matrix Λ = Σ^{-1}. We consider the following block matrix view of Λ:
    Λ = λ11 λ12
        λ21 λ22
    The block matrix view of the Cholesky factor L is:
    L = l11  0
        l21 l22
    such that Λ = LL^T. In order for this factorization to be unique, the diagonal entries need to be positive. For
    this we apply an exponential function to the diagonal entries.

    The conditional mean of the Gaussians given an input x is computed as:
    μ_o - λ11^{-1} λ12 (x - μ_i)

    Hence, since only λ11 = l11 l11^T and λ12 = l11 l21^T are needed, we only store l11 and l21. Moreover, the
    computation of the conditional mean can be simplified further:
    μ_o - (l11 l11^T)^{-1} l11 l21^T (x - μ_i) = μ_o - l11^{-T} l21^T (x - μ_i)

    The uncertainty of the Gaussians is computed as:
    u = log det Σ_{o | i} = -log det λ11 = -2 * log det l11 = -2 * tr log l11

    Args:
        num_gaussians: number of Gaussians
        in_features: size of each input sample
        out_features: size of each output sample

    Shape:
        - Input: if the shape is (in_features,), the input is broadcast to (num_gaussians, in_features). If the shape is
          (B, in_features), the input is broadcast to (B, num_gaussians, in_features). If the shape is
          (B, num_gaussians, in_features), the inputs are mapped to the corresponding Gaussians. B denotes the batch
          size.
        - Output: Conditional mean takes the shape(num_gaussians, out_features) or (B, num_gaussians, out_features)
          depending on the input shape. Uncertainty has shape (num_gaussians,) or (B, num_gaussians,).

    Attributes:
        mean: the means of the Gaussians. The shape is (num_gaussians, out_features + in_features).
        l11: the upper left block of the Cholesky factor of the precision matrix. The shape is
             (num_gaussians, out_features, out_features).
        l21: the bottom left block of the Cholesky factor of the precision matrix. The shape is
             (num_gaussians, in_features, out_features).

    """

    def __init__(
        self, num_gaussians: int, in_features: int, out_features: int, init_output: torch.Tensor = None
    ) -> None:
        super().__init__()

        self.num_gaussians = num_gaussians
        self.in_features = in_features
        self.out_features = out_features

        self.mean = nn.Parameter(torch.empty(num_gaussians, out_features + in_features))
        self.l11 = nn.Parameter(torch.empty(num_gaussians, out_features, out_features))
        self.l21 = nn.Parameter(torch.empty(num_gaussians, in_features, out_features))

        self.tril_kwargs = dict(diagonal=-1)
        self.diag_kwargs = dict(offset=0, dim1=-2, dim2=-1)

        self.reset_parameters(init_output)

    def reset_parameters(self, init_output: torch.Tensor = None) -> None:
        if init_output is None:
            nn.init.zeros_(self.mean)
        else:
            self.mean.data[:, : self.out_features] = init_output
        nn.init.zeros_(self.l11)
        nn.init.zeros_(self.l21)

    def forward(self, x: torch.Tensor, return_l11: bool = False) -> HGSOutput:
        if x.dim() == 1:
            x = x.repeat(self.num_gaussians, 1)
        elif x.dim() == 2:
            x = x.unsqueeze(1).repeat(1, self.num_gaussians, 1)

        dx = x - self.mean[:, self.out_features :]
        dx = dx[..., None]

        l11 = self.l11.tril(**self.tril_kwargs) + torch.diag_embed(
            torch.exp(self.l11.diagonal(**self.diag_kwargs)), **self.diag_kwargs
        )
        l21 = self.l21

        cond_mean = (
            self.mean[:, : self.out_features] - torch.linalg.solve_triangular(l11.mT, l21.mT @ dx, upper=True)[..., 0]
        )

        uncertainty = -2 * self.l11.diagonal(**self.diag_kwargs).sum(dim=-1).unsqueeze(0).repeat(x.shape[0], 1)

        if return_l11:
            return cond_mean, uncertainty, l11

        return cond_mean, uncertainty