import numpy as np
import torch
import nibabel as nib
import yaml
from pathlib import Path
from scipy.special import sph_harm
from numpy.typing import ArrayLike


def parse_cfg(cfg_path: Path) -> dict[any, any]:
    """
    Parses the yaml file to a dictionary
    Args:
        cfg_path (Path): Path instance that points to the configuration yaml file
    Returns:
        dict[any, any]: parsed yaml file
    """
    with open(cfg_path, "r", encoding="utf8") as cfgyaml:
        cfg = yaml.safe_load(cfgyaml)
    return cfg


def parse_response(path: Path) -> np.array:
    with open(path, "r") as f:
        lines = f.readlines()
    lines = [line for line in lines if line[0] != "#"]
    values = [np.array(line.split(), dtype=float) for line in lines]

    return np.concatenate([values], axis=1)


def parse_mrtrix(path: Path) -> np.array:
    with open(path, "r") as f:
        lines = f.readlines()
    lines = [line for line in lines if line[0] != "#"]
    values = [np.array(line.split(), dtype=float) for line in lines]

    return np.concatenate([values], axis=1)


def parse_bvecs(path: Path) -> np.array:
    with open(path, "r") as f:
        x, y, z = f.readlines()

    x = np.array(x.split(), dtype=float)
    y = np.array(y.split(), dtype=float)
    z = np.array(z.split(), dtype=float)

    return np.stack([x, y, z], axis=1)


def parse_bvecs_col(path: Path) -> np.array:
    with open(path, "r") as f:
        strings = f.readlines()

    vec_list = [string.strip().split() for string in strings]

    return np.array(vec_list, dtype=float)


def parse_bvals(path: Path) -> np.array:
    with open(path, "r") as f:
        x = f.readline()

    return np.array([int(float(value)) for value in x.strip().split()], dtype=int)


def save_bvecs(bvecs: np.array, output_path: Path) -> None:
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)
    bvecs = bvecs.astype(str)
    with open(output_path, "w") as f:
        f.writelines(
            [
                " ".join(bvecs[:, 0]),
                "\n",
                " ".join(bvecs[:, 1]),
                "\n",
                " ".join(bvecs[:, 2]),
            ]
        )


def save_bvals(bvals: np.array, output_path: Path) -> None:
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)
    bvecs = bvals.astype(str)
    with open(output_path, "w") as f:
        f.writelines(
            [
                " ".join(bvecs)
            ]
        )


def select_bvecs(bvec_path: Path, output_path: Path, selected_grads: ArrayLike) -> None:
    with open(bvec_path, "r") as f:
        x, y, z = f.readlines()

    sel_x = [val for i, val in enumerate(x.split()) if i in selected_grads]
    new_x = " ".join(sel_x)

    sel_y = [val for i, val in enumerate(y.split()) if i in selected_grads]
    new_y = " ".join(sel_y)

    sel_z = [val for i, val in enumerate(z.split()) if i in selected_grads]
    new_z = " ".join(sel_z)

    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    with open(output_path, "w") as f:
        f.writelines([new_x, "\n", new_y, "\n", new_z])


def flip_bvecs(bvec_path: Path, output_path: Path, axis: tuple[int, int, int]) -> None:
    with open(bvec_path, "r") as f:
        x, y, z = f.readlines()

    x = np.array(x.split(), dtype=float) * axis[0]
    y = np.array(y.split(), dtype=float) * axis[1]
    z = np.array(z.split(), dtype=float) * axis[2]

    new_x = " ".join(x.astype(str))
    new_y = " ".join(y.astype(str))
    new_z = " ".join(z.astype(str))

    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    with open(output_path, "w") as f:
        f.writelines([new_x, "\n", new_y, "\n", new_z])


def select_bvals(bval_path: Path, output_path: Path, selected_grads: ArrayLike) -> None:
    with open(bval_path, "r") as f:
        x = f.readline()

    sel_x = [val for i, val in enumerate(x.split()) if i in selected_grads]
    new_x = " ".join(sel_x)

    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    with open(output_path, "w") as f:
        f.write(new_x)


def select_dwi_img(
    dwi_path: Path, output_path: Path, selected_grads: ArrayLike
) -> None:
    nifti_img = nib.load(dwi_path)
    img_data = nifti_img.get_fdata()[..., selected_grads]

    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    sel_img = nib.Nifti1Image(
        img_data, affine=nifti_img.affine, header=nifti_img.header
    )
    nib.save(sel_img, output_path)


def spherical_to_cartesian(spherical_vec: np.array) -> np.array:
    rho = spherical_vec[..., 0]
    phi = spherical_vec[..., 1]
    theta = spherical_vec[..., 2]

    return np.stack(
        [
            rho * np.sin(phi) * np.cos(theta),
            rho * np.sin(phi) * np.sin(theta),
            rho * np.cos(phi),
        ],
        axis=-1,
    )


def spherical_to_cartesian_tensor(spherical_vec: torch.Tensor) -> torch.Tensor:
    rho = spherical_vec[..., 0]
    phi = spherical_vec[..., 1]
    theta = spherical_vec[..., 2]

    return torch.stack(
        [
            rho * torch.sin(phi) * torch.cos(theta),
            rho * torch.sin(phi) * torch.sin(theta),
            rho * torch.cos(phi),
        ],
        axis=-1,
    )


def cartesian_to_spherical(cartesian_vec: np.array) -> np.array:
    # (rho, phi, theta)
    # rho -> radial distance [0, inf)
    # phi -> elevation [0, pi]
    # theta -> azimuth [0, 2pi)

    x = cartesian_vec[..., 0]
    y = cartesian_vec[..., 1]
    z = cartesian_vec[..., 2]

    x2y2 = x**2 + y**2

    r = np.sqrt(x2y2 + z**2)
    phi = np.arccos((z / r))
    theta = np.arctan2(y, x)
    theta = np.where(theta < 0, theta + 2 * np.pi, theta)

    return np.stack([r, phi, theta], axis=-1)


def cartesian_to_spherical_tensor(cartesian_vec: torch.Tensor) -> torch.Tensor:
    # (rho, phi, theta)
    # rho -> radial distance [0, inf)
    # phi -> elevation [0, pi]
    # theta -> azimuth [0, 2pi)

    x = cartesian_vec[..., 0]
    y = cartesian_vec[..., 1]
    z = cartesian_vec[..., 2]

    x2y2 = x**2 + y**2

    r = torch.clamp(torch.sqrt(x2y2 + z**2), 0.000001)
    z_r = torch.clamp((z / r), -1, 1)
    phi = torch.arccos(z_r)
    theta = torch.arctan2(y, x)
    theta = torch.where(theta < 0, theta + 2 * torch.pi, theta)

    return torch.stack([r, phi, theta], axis=-1)


def sum_harmonics(
    coeffs: torch.tensor, lmax: int, thetas: np.array, phis: np.array
) -> torch.tensor:
    # MRtrix
    device = coeffs.device
    diff_values = torch.zeros(coeffs.shape[0]).to(device)
    coeff_idx = 0

    for i in range(0, lmax + 1, 2):
        for m in range(-i, i + 1):
            if m < 0:
                value = sph_harm(-m, i, thetas, phis)
                diff_values += coeffs[:, coeff_idx] * torch.tensor(
                    value.imag * 2**0.5
                ).to(device)
            if m > 0:
                value = sph_harm(m, i, thetas, phis)
                diff_values += coeffs[:, coeff_idx] * torch.tensor(
                    value.real * 2**0.5
                ).to(device)
            else:
                value = sph_harm(m, i, thetas, phis)
                diff_values += coeffs[:, coeff_idx] * torch.tensor(value.real).to(
                    device
                )

            coeff_idx += 1

    return diff_values


def precompute_harmonics(lmax: int, thetas: np.array, phis: np.array) -> torch.tensor:
    n_coeff = (lmax + 1) * (lmax + 2) // 2
    sh_constants = np.zeros((thetas.shape[0], n_coeff))
    j = 0
    for i in range(0, lmax + 1, 2):
        for m in range(-i, i + 1):
            if m < 0:
                value = sph_harm(-m, i, thetas, phis)
                sh_constants[:, j] = value.imag * 2**0.5
            if m > 0:
                value = sph_harm(m, i, thetas, phis)
                sh_constants[:, j] = value.real * 2**0.5
            else:
                value = sph_harm(m, i, thetas, phis)
                sh_constants[:, j] = value.real
            j += 1
    return np.array(sh_constants)
