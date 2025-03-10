import torch.nn as nn
import torch
import wandb
from functools import partial


def positional_encoding(
    x: torch.Tensor, lpos: int, sigma: float = None
) -> torch.Tensor:
    if not sigma:
        js = 2 ** torch.arange(lpos).to(x.device) * torch.pi
    else:
        js = 2 ** torch.linspace(0, sigma, lpos).to(x.device) * torch.pi

    jx = torch.einsum("ix, j -> ijx", x, js)
    sin_out = torch.sin(jx).reshape(x.shape[0], -1)
    cos_out = torch.cos(jx).reshape(x.shape[0], -1)
    return torch.cat([x, sin_out, cos_out], dim=-1)


def input_mapping(x, B=None):
    if B is None:
        return x
    else:
        x_proj = (2.0 * torch.pi * x) @ B.T
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


def progressive_emb(x_emb: torch.Tensor, t_frac: float) -> torch.Tensor:
    a = torch.ones(x_emb.shape[1]).to(x_emb.device)
    start = int(t_frac * x_emb.shape[1] + 3)
    end = int(t_frac * x_emb.shape[1] + 4)
    a[start:end] = (t_frac * x_emb.shape[1]) - int(t_frac * x_emb.shape[1])
    a[int(end) :] = 0

    return x_emb * a.unsqueeze(dim=0)


def histogram_hook(name):
    def hook(_, _input, output):
        with torch.no_grad():
            wandb.log(
                {
                    f"{name}_input": wandb.Histogram(
                        _input[0].detach().cpu(), num_bins=20
                    )
                }
            )
            wandb.log(
                {f"{name}_output": wandb.Histogram(output.detach().cpu(), num_bins=20)}
            )

    return hook


class Fod_NeSH(nn.Module):
    def __init__(
        self,
        l_max=8,
        lpos=10,
        hidden_dim=256,
        n_layers=11,
        sigma=None,
        gaussian=True,
    ) -> None:
        super().__init__()

        output_size = (l_max + 1) * (l_max + 2) // 2

        input_size = lpos * 2 if gaussian else lpos * 6 + 3
        self.mlp = nn.Sequential(
            *(
                [nn.Linear(input_size, hidden_dim), nn.ReLU()]
                + [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()] * (n_layers - 1)
                + [nn.Linear(hidden_dim, output_size)]
            )
        )

        self.Lpos = lpos
        self.sigma = sigma
        self.B = nn.Parameter(torch.randn([lpos, 3]) * sigma) if gaussian else None

    def forward(self, x, t_frac=None) -> torch.Tensor:
        if self.B is not None:
            x_emb = input_mapping(x, self.B.to(x.device))
        else:
            x_emb = positional_encoding(x, self.Lpos, self.sigma)

        if t_frac is not None:
            x_emb = progressive_emb(x_emb, t_frac)

        return self.mlp(x_emb)


class Fod_NeSH2(nn.Module):
    def __init__(
        self,
        l_max=8,
        lpos=10,
        hidden_dim=256,
        n_layers=11,
        sigma=None,
        gaussian=True,
    ) -> None:
        super().__init__()

        output_size = (l_max + 1) * (l_max + 2) // 2

        input_size = lpos * 2 if gaussian else lpos * 6 + 3
        self.mlp = nn.Sequential(
            *(
                [nn.Linear(input_size, hidden_dim), nn.ReLU()]
                + [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()] * (n_layers - 1)
                + [nn.Linear(hidden_dim, output_size)]
            )
        )

        self.Lpos = lpos
        self.sigma = sigma
        self.B = nn.Parameter(torch.randn([lpos, 3]) * sigma) if gaussian else None

    def forward(self, x, t_frac=None) -> torch.Tensor:
        if self.B is not None:
            x_emb = input_mapping(x, self.B.to(x.device))
        else:
            x_emb = positional_encoding(x, self.Lpos, self.sigma)

        if t_frac is not None:
            x_emb = progressive_emb(x_emb, t_frac)

        return self.mlp(x_emb)


class Split_Fod_NeSH(nn.Module):
    def __init__(
        self,
        l_max=8,
        lpos=10,
        hidden_dim=256,
        n_layers=11,
        sigma=None,
        gaussian=True,
    ) -> None:
        super().__init__()

        output_size = (l_max + 1) * (l_max + 2) // 2
        input_size = lpos * 2 if gaussian else lpos * 6 + 3

        self.mlp_fod = nn.Sequential(
            *(
                [nn.Linear(input_size, hidden_dim), nn.ReLU()]
                + [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()] * (n_layers - 2)
                + [nn.Linear(hidden_dim, output_size)]
            )
        )
        self.mlp_gm = nn.Sequential(
            *(
                [nn.Linear(input_size, hidden_dim), nn.ReLU()]
                + [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()] * (n_layers - 2)
                + [nn.Linear(hidden_dim, 1), nn.Softplus()]
            )
        )
        self.mlp_csf = nn.Sequential(
            *(
                [nn.Linear(input_size, hidden_dim), nn.ReLU()]
                + [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()] * (n_layers - 2)
                + [nn.Linear(hidden_dim, 1), nn.Softplus()]
            )
        )

        self.Lpos = lpos
        self.sigma = sigma
        self.B = nn.Parameter(
            torch.randn([lpos, 3]) * sigma if gaussian else None, requires_grad=False
        )

    def forward(self, x, t_frac=None) -> torch.Tensor:
        if self.B is not None:
            x_emb = input_mapping(x, self.B.to(x.device))
        else:
            x_emb = positional_encoding(x, self.Lpos, self.sigma)

        if t_frac is not None:
            x_emb = progressive_emb(x_emb, t_frac)

        return torch.cat(
            [self.mlp_fod(x_emb), self.mlp_gm(x_emb), self.mlp_csf(x_emb)], dim=-1
        )


class Multi_Fod_NeSH(nn.Module):
    def __init__(
        self,
        l_max=8,
        lpos=10,
        hidden_dim=256,
        n_layers=11,
        sigma=None,
        gaussian=True,
    ) -> None:
        super().__init__()

        output_size = (l_max + 1) * (l_max + 2) // 2
        input_size = lpos * 2 if gaussian else lpos * 6 + 3

        self.mlp = nn.Sequential(
            *(
                [nn.Linear(input_size, hidden_dim), nn.ReLU()]
                + [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()] * (n_layers - 2)
                + [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
            )
        )
        self.fod_head = nn.Linear(hidden_dim, output_size)
        self.gm_head = nn.Sequential(*[nn.Linear(hidden_dim, 1), nn.Softplus()])
        self.csf_head = nn.Sequential(*[nn.Linear(hidden_dim, 1), nn.Softplus()])

        self.Lpos = lpos
        self.sigma = sigma
        self.B = nn.Parameter(torch.randn([lpos, 3]) * sigma) if gaussian else None

    def forward(self, x, t_frac=None) -> torch.Tensor:
        if self.B is not None:
            x_emb = input_mapping(x, self.B.to(x.device))
        else:
            x_emb = positional_encoding(x, self.Lpos, self.sigma)

        if t_frac is not None:
            x_emb = progressive_emb(x_emb, t_frac)

        x_mlp = self.mlp(x_emb)
        return torch.cat(
            [self.fod_head(x_mlp), self.gm_head(x_mlp), self.csf_head(x_mlp)], dim=-1
        )


def create_std_model(cfg: dict, model: nn.Module) -> nn.Module:
    train_cfg = cfg["train_cfg"]
    sigma = train_cfg["sigma"]
    lpos = train_cfg["lpos"]
    hidden_dim = train_cfg["hidden_dim"]
    n_layers = train_cfg["n_layers"]
    l_max = train_cfg["lmax"]

    return model(
        l_max=l_max,
        lpos=lpos,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        sigma=sigma,
        gaussian=cfg["gaussian_encoding"],
    )


MODELS = {
    "fod": partial(create_std_model, model=Fod_NeSH),
    "multishell": partial(create_std_model, model=Multi_Fod_NeSH),
    "split_multi": partial(create_std_model, model=Split_Fod_NeSH),
}


def get_model(cfg: dict) -> nn.Module:
    constructor = MODELS.get(cfg["model_name"], None)
    if constructor is None:
        raise Exception("Model name not recognized")
    return constructor(cfg)
