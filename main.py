import wandb
import torch
import nibabel as nib

from utils import parse_cfg

from loss_functions import get_loss_function
from output_calculators import get_output_calculator
from ml_utils import Trainer
from models import get_model
from datasets import get_dataset
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR


def initialize_run():
    cfg = parse_cfg(Path("configs/example_config_multi.yaml"))
    #wandb.init(project=cfg["project_name"], job_type="testing", config=cfg)

    #cfg = wandb.config
    train_cfg = cfg["train_cfg"]

    width = cfg["width"]
    height = cfg["height"]
    depth = cfg["depth"]

    l_max = train_cfg["lmax"]
    lr = train_cfg["lr"]
    lambda_ = train_cfg["lambda"]
    log_freq = cfg["log_freq"]

    model = get_model(cfg)
    print(model)

    dataset = get_dataset(cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=3,
        drop_last=True,
    )

    loss_fn = get_loss_function(cfg)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.99) if cfg.get("scheduler", False) else None

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)
    epochs = train_cfg["epochs"]

    output_calculator = get_output_calculator(cfg, dataset=dataset, device=device)

    trainer = Trainer(
        model=model,
        dataset=dataset,
        dataloader=dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        l_max=l_max,
        data_shape=(width, height, depth),
        output_calculator=output_calculator,
        log_freq=log_freq,
        lambda_=lambda_,
        slice_id=train_cfg["slice_id"],
        scheduler=scheduler,
    )

    trainer.train()

    file_inf = "test"

    output_folder = Path(cfg["paths"]["output"])
    if not output_folder.exists():
        output_folder.mkdir(parents=True)
    torch.save(trainer.model.state_dict(), output_folder / f"model_{file_inf}.pt")

    if cfg["model_name"] in ["multishell", "split_multi"]:
        (
            nifti_img,
            grad_img,
            coeff_image,
            gm_coeff,
            csf_coeff,
        ) = trainer.create_full_output_image(cfg, dataset.get_scale())

        gm_coeff_img = nib.Nifti1Image(
            gm_coeff, affine=nifti_img.affine, header=nifti_img.header
        )
        nib.save(gm_coeff_img, output_folder / f"gm_coeffs_{file_inf}.nii.gz")

        csf_coeff_img = nib.Nifti1Image(
            csf_coeff, affine=nifti_img.affine, header=nifti_img.header
        )
        nib.save(csf_coeff_img, output_folder / f"csf_coeffs_{file_inf}.nii.gz")
    else:
        nifti_img, grad_img, coeff_image = trainer.create_full_output_image(
            cfg, dataset.get_scale()
        )

    full_nifti_img = nib.Nifti1Image(
        grad_img, affine=nifti_img.affine, header=nifti_img.header
    )
    nib.save(full_nifti_img, output_folder / f"grads_{file_inf}.nii.gz")

    full_coeff_img = nib.Nifti1Image(
        coeff_image, affine=nifti_img.affine, header=nifti_img.header
    )
    nib.save(full_coeff_img, output_folder / f"coeffs_{file_inf}.nii.gz")


if __name__ == "__main__":
    initialize_run()
