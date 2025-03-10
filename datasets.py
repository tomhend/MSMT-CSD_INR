from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import parse_bvecs, parse_bvals, parse_response, parse_mrtrix

def create_input_space(width: int, height: int, depth: int) -> torch.Tensor:
    p_width = 2 / width
    p_height = 2 / height
    p_depth = 2 / depth

    x = torch.linspace(-1 + p_width / 2, (1 - p_width / 2), width)
    y = torch.linspace(-1 + p_height / 2, (1 - p_height / 2), height)
    z = torch.linspace(-1 + p_depth / 2, (1 - p_depth / 2), depth)

    input_tensor = torch.cartesian_prod(x, y, z)

    return input_tensor.reshape(width, height, depth, 3)


def get_dwi_indices(bvals: np.array, bval: float, delta: float):
    bval_low = bval - delta
    bval_high = bval + delta

    return np.nonzero((bval_low < bvals) & (bvals < bval_high))[0]


def get_mean_b0(img: np.array, b0_idx: np.array):
    if len(b0_idx) == 0:
        raise Exception("No b0 images found")
    b0_imgs = img[..., b0_idx]
    if b0_idx.sum() == 1:
        return b0_imgs
    return b0_imgs.mean(axis=-1, keepdims=True)

class DiffusionDataset(Dataset):
    def get_dwi_idx(self) -> np.ndarray:
        pass

    def get_bvals(self) -> np.ndarray:
        pass

    def get_scale(self) -> float | np.ndarray:
        pass

    def get_response(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    def get_directions(self) -> np.ndarray:
        pass

class SingleShellDataset(DiffusionDataset):
    def __init__(
        self,
        bvec_path: Path,
        bval_path: Path,
        mrtrix_bvec_path: Path,
        response_path: Path,
        shell: int,
        bval_delta: int,
        nifti_path: Path,
        mask_path: Path = None,
        scale: bool = True,
    ) -> None:
        nifti_file = nib.load(nifti_path)

        if bvec_path and bval_path:
            self.bvals = parse_bvals(bval_path)
            self.bvecs = parse_bvecs(bvec_path)
        else:
            mrtrix_bvecs = parse_mrtrix(mrtrix_bvec_path)
            self.bvals = mrtrix_bvecs[:, -1]
            self.bvecs = mrtrix_bvecs[:, :3]

        full_img = nifti_file.get_fdata()
        self.dwi_idx = (self.bvals > (shell - bval_delta)) & (self.bvals < (shell + bval_delta))
        output_array = full_img[..., self.dwi_idx]  # remove b0
        width, height, depth, n_grad = output_array.shape

        self.cart_bvecs = parse_bvecs(bvec_path)[self.dwi_idx]  # remove b0
        # Separate into function for generating input coordinates
        self.input_tensor = create_input_space(width, height, depth)

        self.scale_value = np.percentile(output_array, 99) if scale else 1
        output_array = output_array / self.scale_value

        if mask_path:
            mask_data = nib.load(mask_path).get_fdata().astype(int)
            brain_idx = np.asarray(mask_data == 1).nonzero()

            self.input_tensor = self.input_tensor[
                brain_idx[0], brain_idx[1], brain_idx[2]
            ]
            output_array = output_array[brain_idx[0], brain_idx[1], brain_idx[2], :]
        else:
            self.input_tensor = self.input_tensor.reshape(width * height * depth, 3)
            output_array = output_array.reshape(width * height * depth, -1)

        self.output_tensor = torch.tensor(output_array, dtype=torch.float32)
        self.response_coeff = (
            torch.tensor(
                parse_response(response_path)[0], dtype=torch.float
            )
            / self.scale_value
        )

    def __len__(self) -> int:
        return self.input_tensor.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_tensor[idx], self.output_tensor[idx]

    def get_dwi_idx(self) -> np.ndarray:
        return self.dwi_idx.nonzero()[0]

    def get_scale(self) -> float | np.ndarray:
        return self.scale_value

    def get_response(self) -> np.ndarray:
        return self.response_coeff

    def get_directions(self) -> np.ndarray:
        return self.cart_bvecs

    def get_bvals(self) -> np.ndarray:
        return self.bvals


class MultiShellDataset(Dataset):
    def __init__(
        self,
        bvec_path: Path,
        bval_path: Path,
        mrtrix_bvec_path: Path,
        response_paths: list[Path],
        bval_delta: int,
        nifti_path: Path,
        shells: np.array,
        mask_path: Path = None,
        scale: bool = True,
    ) -> None:
        nifti_file = nib.load(nifti_path)
        nifti_img = nifti_file.get_fdata()

        if bvec_path and bval_path:
            bvals = parse_bvals(bval_path)
            all_bvecs = parse_bvecs(bvec_path)
        else:
            mrtrix_bvecs = parse_mrtrix(mrtrix_bvec_path)
            bvals = mrtrix_bvecs[:, -1]
            all_bvecs = mrtrix_bvecs[:, :3]

        self.shells = shells

        used = np.array([bval in shells for bval in bvals]) # used
        self.dwi_idx = used
        scale_values = [np.percentile(nifti_img[..., get_dwi_indices(bvals, bval, bval_delta)], 99) if scale else 1 for bval in bvals]

        b0_idx = (bvals < bval_delta)
        n_b0 = b0_idx.sum()
        if n_b0 > 0:
            b0_img = get_mean_b0(nifti_img, b0_idx)
            b0_scale = np.percentile(b0_img, 99) if scale else 1

            used[b0_idx] = False
            used = np.append(used, True)
            bvals = np.append(bvals, 0)
            scale_values = np.append(scale_values, b0_scale)

            nifti_img = np.concatenate([nifti_img, get_mean_b0(nifti_img, b0_idx)], axis=-1)
            all_bvecs = np.concatenate([all_bvecs, np.array([1, 0, 0])[None, :]])

        full_img = nifti_img[..., used]
        self.sel_bvals = bvals[used]
        self.scale_value = scale_values[used]

        output_array = full_img/self.scale_value

        self.cart_bvecs = all_bvecs[used]

        width, height, depth, n_grad = output_array.shape
        self.input_tensor = create_input_space(width, height, depth)

        if mask_path:
            mask_data = nib.load(mask_path).get_fdata().astype(int)
            brain_idx = np.asarray(mask_data == 1).nonzero()
            self.input_tensor = self.input_tensor[
                brain_idx[0], brain_idx[1], brain_idx[2]
            ]
            output_array = output_array[brain_idx[0], brain_idx[1], brain_idx[2]]
        else:
            self.input_tensor = self.input_tensor.reshape(width * height * depth, 3)
            output_array = output_array.reshape(width * height * depth, -1)

        self.output_tensor = torch.tensor(output_array, dtype=torch.float)

        resp_scaler = np.array([scale_values[(bvals == shell)&used][0] for shell in shells])[:, None]
        self.wm_response = torch.tensor(
            parse_response(response_paths[0]) / resp_scaler,
            dtype=torch.float,
        )
        self.gm_response = torch.tensor(
            parse_response(response_paths[1]) / resp_scaler,
            dtype=torch.float,
        )
        self.csf_response = torch.tensor(
            parse_response(response_paths[2]) / resp_scaler,
            dtype=torch.float,
        )

    def __len__(self) -> int:
        return self.input_tensor.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_tensor[idx], self.output_tensor[idx]

    def get_dwi_idx(self) -> np.ndarray:
        return self.dwi_idx.nonzero()[0]

    def get_bvals(self) -> np.ndarray:
        return self.sel_bvals

    def get_scale(self) -> float | np.ndarray:
        return self.scale_value

    def get_response(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.wm_response, self.gm_response, self.csf_response

    def get_directions(self) -> np.ndarray:
        return self.cart_bvecs



def create_singleshell(cfg: dict) -> DiffusionDataset:
    mask_path = Path(cfg["paths"]["mask"]) if cfg["paths"].get("mask", None) else None
    bvec_path = Path(cfg["paths"]["fsl_bvecs"]) if cfg["paths"].get("fsl_bvecs", None) else None
    bval_path = Path(cfg["paths"]["fsl_bvals"]) if cfg["paths"].get("fsl_bvals", None) else None
    mrtrix_bvec_path = Path(cfg["paths"]["mrtrix_bvecs"]) if cfg["paths"].get("mrtrix_bvecs", None) else None

    response_path = Path(cfg['paths']['response'])
    dataset = SingleShellDataset(
        bvec_path=bvec_path,
        bval_path=bval_path,
        mrtrix_bvec_path=mrtrix_bvec_path,
        response_path=response_path,
        shell=cfg["shells"][0],
        bval_delta=cfg["bval_delta"],
        nifti_path=Path(cfg["paths"]["nifti"]),
        mask_path=mask_path,
        scale=cfg["scale_data"],
    )

    return dataset


def create_multishell(cfg: dict) -> MultiShellDataset:
    mask_path = Path(cfg["paths"]["mask"]) if cfg["paths"].get("mask", None) else None
    bvec_path = Path(cfg["paths"]["fsl_bvecs"]) if cfg["paths"].get("fsl_bvecs", None) else None
    bval_path = Path(cfg["paths"]["fsl_bvals"]) if cfg["paths"].get("fsl_bvals", None) else None
    mrtrix_bvec_path = Path(cfg["paths"]["mrtrix_bvecs"]) if cfg["paths"].get("mrtrix_bvecs", None) else None

    response_paths = [
        Path(cfg["paths"]["wm_response"]),
        Path(cfg["paths"]["gm_response"]),
        Path(cfg["paths"]["csf_response"]),
    ]
    dataset = MultiShellDataset(
        bvec_path=bvec_path,
        bval_path=bval_path,
        mrtrix_bvec_path=mrtrix_bvec_path,
        response_paths=response_paths,
        bval_delta=cfg["bval_delta"],
        shells=np.array(cfg["shells"]),
        nifti_path=Path(cfg["paths"]["nifti"]),
        mask_path=mask_path,
        scale=cfg["scale_data"],
    )

    return dataset

DATASETS = {
    "singleshell": create_singleshell,
    "multishell": create_multishell,
}


def get_dataset(cfg: dict) -> Dataset:
    constructor = DATASETS.get(cfg["dataset_name"], None)
    if constructor is None:
        raise Exception("Dataset name not recognized")
    return constructor(cfg)
