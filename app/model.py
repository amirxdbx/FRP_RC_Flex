from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


EPS = 1e-9
EPS_CU = 0.003
EF_IS_GPA = True


class PINNFRP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        h1: int = 48,
        h2: int = 24,
        gate_h1: int = 36,
        gate_h2: int = 18,
        dropout: float = 0.0,
        use_failure_head: bool = True,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        self.head_m_pred_cr = nn.Linear(h2, 1)
        self.head_m_pred_fr = nn.Linear(h2, 1)
        self.head_cd_cr = nn.Sequential(nn.Linear(h2, 1), nn.Sigmoid())
        self.head_cd_fr = nn.Sequential(nn.Linear(h2, 1), nn.Sigmoid())

        self.gate_net = None
        if use_failure_head:
            self.gate_net = nn.Sequential(
                nn.Linear(in_dim, gate_h1),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(gate_h1, gate_h2),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(gate_h2, 1),
            )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        h = self.net(x)
        m_pred_cr = self.head_m_pred_cr(h)
        m_pred_fr = self.head_m_pred_fr(h)
        cd_cr_raw = self.head_cd_cr(h)
        cd_fr_raw = self.head_cd_fr(h)
        pfr_logits = self.gate_net(x) if self.gate_net is not None else None
        return m_pred_cr, m_pred_fr, cd_cr_raw, cd_fr_raw, pfr_logits


@dataclass(frozen=True)
class PhysicsOutputs:
    c_cr: torch.Tensor
    a_cr: torch.Tensor
    C_cr: torch.Tensor
    T_cr: torch.Tensor
    Mn_cr: torch.Tensor
    cd_cr: torch.Tensor
    stress_ratio_cr_uncapped: torch.Tensor
    alpha_cr: torch.Tensor
    beta_cr: torch.Tensor
    c_fr: torch.Tensor
    a_fr: torch.Tensor
    C_fr: torch.Tensor
    T_fr: torch.Tensor
    Mn_fr: torch.Tensor
    cd_fr: torch.Tensor
    eps_c_fr: torch.Tensor
    alpha_fr: torch.Tensor
    beta_fr: torch.Tensor


def compute_aci_terms_from_raw_two_cd(
    b: torch.Tensor,
    d: torch.Tensor,
    fc: torch.Tensor,
    Af: torch.Tensor,
    Ef: torch.Tensor,
    ffu: torch.Tensor,
    cd_cr_raw: torch.Tensor,
    cd_fr_raw: torch.Tensor,
    cd_min: float,
    cd_max: float,
) -> PhysicsOutputs:
    Ef_mpa = Ef * (1000.0 if EF_IS_GPA else 1.0)
    Ec_mpa = 4700.0 * torch.sqrt(fc)
    eps_c_prime = 1.7 * fc / (Ec_mpa + EPS)

    u_cr = cd_cr_raw.squeeze(1)
    u_fr = cd_fr_raw.squeeze(1)

    cd_cr_min = (Ef_mpa * EPS_CU) / (ffu + Ef_mpa * EPS_CU + EPS)
    cd_cr_min = torch.clamp(cd_cr_min, cd_min, cd_max)
    cd_cr = cd_cr_min + (cd_max - cd_cr_min) * u_cr
    cd_cr = torch.clamp(cd_cr, cd_min, cd_max)

    eps_f_fr = ffu / (Ef_mpa + EPS)
    cd_fr_max = EPS_CU / (eps_f_fr + EPS_CU + EPS)
    cd_fr_max = torch.clamp(cd_fr_max, cd_min, cd_max)
    cd_fr = cd_min + (cd_fr_max - cd_min) * u_fr
    cd_fr = torch.clamp(cd_fr, cd_min, cd_max)

    c_cr = cd_cr * d
    c_fr = cd_fr * d

    eps_f_cr = EPS_CU * (d - c_cr) / (c_cr + EPS)
    f_unc_cr = Ef_mpa * eps_f_cr
    stress_ratio_cr_uncapped = f_unc_cr / (ffu + EPS)
    f_cap_cr = torch.minimum(f_unc_cr, ffu)
    T_cr = Af * f_cap_cr

    beta_cr = (4.0 * eps_c_prime - EPS_CU) / (6.0 * eps_c_prime - 2.0 * EPS_CU + EPS)
    beta_cr = torch.clamp(beta_cr, 0.05, 0.95)
    alpha_cr = (3.0 * EPS_CU * eps_c_prime - EPS_CU**2) / (3.0 * beta_cr * (eps_c_prime**2) + EPS)
    a_cr = beta_cr * c_cr
    C_cr = alpha_cr * fc * b * a_cr
    Mn_cr = T_cr * (d - 0.5 * a_cr) / 1e6

    T_fr = Af * ffu
    eps_c_fr = eps_f_fr * c_fr / (d - c_fr + EPS)
    eps_c_fr = torch.clamp(eps_c_fr, 0.0, EPS_CU)

    beta_fr = (4.0 * eps_c_prime - eps_c_fr) / (6.0 * eps_c_prime - 2.0 * eps_c_fr + EPS)
    beta_fr = torch.clamp(beta_fr, 0.05, 0.95)
    alpha_fr = (3.0 * eps_c_fr * eps_c_prime - eps_c_fr**2) / (3.0 * beta_fr * (eps_c_prime**2) + EPS)
    a_fr = beta_fr * c_fr
    C_fr = alpha_fr * fc * b * a_fr
    Mn_fr = T_fr * (d - 0.5 * a_fr) / 1e6

    return PhysicsOutputs(
        c_cr=c_cr,
        a_cr=a_cr,
        C_cr=C_cr,
        T_cr=T_cr,
        Mn_cr=Mn_cr,
        cd_cr=cd_cr,
        stress_ratio_cr_uncapped=stress_ratio_cr_uncapped,
        alpha_cr=alpha_cr,
        beta_cr=beta_cr,
        c_fr=c_fr,
        a_fr=a_fr,
        C_fr=C_fr,
        T_fr=T_fr,
        Mn_fr=Mn_fr,
        cd_fr=cd_fr,
        eps_c_fr=eps_c_fr,
        alpha_fr=alpha_fr,
        beta_fr=beta_fr,
    )

