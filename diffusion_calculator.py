import math
from functools import lru_cache

import numpy as np
import torch
from dipy.data import get_sphere
from scipy.special import sph_harm
from utils import (
    cartesian_to_spherical,
    spherical_to_cartesian,
)


def create_y_mat(thetas: np.array, phis: np.array, l_max: int) -> torch.Tensor:
    n_dir = thetas.shape[0]
    n = (l_max + 1) * (l_max + 2) // 2

    y_mat = torch.zeros((n_dir, n))
    for l in range(0, l_max + 1, 2):
        for m in range(-l, l + 1):
            coef_idx = (l**2 + l) // 2 + m
            Y = sph_harm(np.abs(m), l, thetas, phis)
            if m < 0:
                y_mat[:, coef_idx] = torch.tensor(Y.imag * np.sqrt(2))
            if m > 0:
                y_mat[:, coef_idx] = torch.tensor(Y.real * np.sqrt(2))
            if m == 0:
                y_mat[:, coef_idx] = torch.tensor(Y.real)

    return y_mat


def get_rescale_value(l_max: int, rescale: bool = True) -> torch.Tensor:
    rescale_value = (
        torch.sqrt(4 * np.pi / (2 * torch.arange(0, l_max + 1, 2) + 1))
        if rescale
        else torch.ones(l_max // 2 + 1)
    )
    return rescale_value


def create_conv_vec(
    l_max: int, resp_coeff: torch.Tensor, rescale: bool = True
) -> torch.Tensor:
    rescale_value = get_rescale_value(l_max, rescale)
    conv_vec = torch.zeros((l_max + 1) * (l_max + 2) // 2)
    for i, l in enumerate(range(0, l_max + 1, 2)):
        for m in range(-l, l + 1):
            coef_idx = (l**2 + l) // 2 + m
            conv_vec[coef_idx] = resp_coeff[i] * rescale_value[i]
    return conv_vec


class SignalSingleShell:
    def __init__(
        self,
        l_max: int,
        resp_coeff: torch.Tensor,
        cart_bvec: np.array = None,
        sph_bvec: np.array = None,
        device: str = "cpu",
        fod_rescale: bool = True,
    ):
        self.l_max = l_max
        self.device = device

        self.y_mat = None

        sphere = get_sphere("repulsion724")  # 100, 200, 724
        sph_sphere_vecs = cartesian_to_spherical(sphere.vertices)
        sphere_thetas = sph_sphere_vecs[:, 2]
        sphere_phis = sph_sphere_vecs[:, 1]
        self.sphere_y_mat = create_y_mat(sphere_thetas, sphere_phis, l_max).to(device)

        if cart_bvec is not None:
            self.cart_bvec = cart_bvec
            sph_bvec = cartesian_to_spherical(cart_bvec)

        if sph_bvec is not None:
            self.sph_bvec = sph_bvec

            if cart_bvec is None:
                self.cart_bvec = spherical_to_cartesian(sph_bvec)

            sample_phis = sph_bvec[..., 1]
            sample_thetas = sph_bvec[..., 2]
            self.y_mat = create_y_mat(sample_thetas, sample_phis, l_max).to(device)

        self.conv_vec = create_conv_vec(l_max, resp_coeff, fod_rescale).to(device)

    def compute_signal_from_coeff(self, coeffs: torch.tensor):
        return torch.einsum("bk, k, dk -> bd", coeffs, self.conv_vec, self.y_mat)

    def compute_negative_signal(self, coeffs: torch.Tensor):
        # return torch.clamp(
        #     torch.einsum("bk, dk -> bd", coeffs, self.sphere_y_mat), max=0
        # )

        amplitudes = torch.einsum("bk, dk -> bd", coeffs, self.sphere_y_mat)
        max_values = torch.mean(amplitudes, dim=1, keepdim=True) * 0.1
        return torch.clamp(amplitudes - max_values, max=0)


class SignalMultishell:
    def __init__(
        self,
        l_max: int,
        resp_coeff: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        bval_idx: list,
        cart_bvec: np.array = None,
        sph_bvec: np.array = None,
        device: str = "cpu",
        fod_rescale: bool = True,
    ):
        self.l_max = l_max
        self.device = device

        self.y_mat = None
        wm_coeff, gm_coeff, csf_coeff = (
            resp_coeff[0],
            resp_coeff[1].to(device),
            resp_coeff[2].to(device),
        )

        self.gm_coeff = torch.stack([gm_coeff[idx] for idx in bval_idx])
        self.csf_coeff = torch.stack([csf_coeff[idx] for idx in bval_idx])

        conv_vecs = [
            create_conv_vec(l_max, coeffs, fod_rescale).to(device)
            for coeffs in wm_coeff
        ]

        self.conv_mat = torch.stack([conv_vecs[idx] for idx in bval_idx])
        self.fod_rescale = fod_rescale

        sphere = get_sphere("repulsion724")  # 100, 200, 724
        sph_sphere_vecs = cartesian_to_spherical(sphere.vertices)
        sphere_thetas = sph_sphere_vecs[:, 2]
        sphere_phis = sph_sphere_vecs[:, 1]
        self.sphere_y_mat = create_y_mat(sphere_thetas, sphere_phis, l_max).to(device)

        if cart_bvec is not None:
            self.cart_bvec = cart_bvec
            sph_bvec = cartesian_to_spherical(cart_bvec)

        if sph_bvec is not None:
            self.sph_bvec = sph_bvec
            if cart_bvec is None:
                self.cart_bvec = spherical_to_cartesian(sph_bvec)

            sample_phis = sph_bvec[..., 1]
            sample_thetas = sph_bvec[..., 2]
            self.y_mat = create_y_mat(sample_thetas, sample_phis, l_max).to(device)

    def compute_signal_from_coeff(self, coeffs: torch.tensor):
        fod_coeffs = coeffs[:, :-2]
        # fod_frac = coeffs[:, -3]
        gm_coeff = coeffs[:, [-2]]
        csf_coeff = coeffs[:, [-1]]

        fod_signal = torch.einsum(
            "bk, dk, dk -> bd", fod_coeffs, self.conv_mat, self.y_mat
        )  # * fod_frac[:, None, None]

        gm_signal = torch.einsum(
            "bk, dk -> bd", gm_coeff, self.gm_coeff
        )
        csf_signal = torch.einsum(
            "bk, dk -> bd", csf_coeff, self.csf_coeff
        )

        return fod_signal + gm_signal + csf_signal

    def compute_negative_signal(self, coeffs: torch.Tensor):
        fod_coeffs = coeffs[:, :-2]
        amplitudes = torch.einsum("bk, dk -> bd", fod_coeffs, self.sphere_y_mat)
        max_values = torch.mean(amplitudes, dim=1, keepdim=True) * 0.1
        return torch.clamp(amplitudes - max_values, max=0)
