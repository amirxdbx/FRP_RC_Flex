from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import pandas as pd
import torch

from app.config import METADATA_PATH, MODEL_PATH
from app.model import PINNFRP, compute_aci_terms_from_raw_two_cd


class Predictor:
    def __init__(self) -> None:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.device = torch.device("cpu")
        self.feature_names: list[str] = self.metadata["feature_names"]
        self.threshold: float = float(self.metadata["gate_threshold"])
        self.eval_gate_mode: str = self.metadata["eval_gate_mode"]
        self.cd_min: float = float(self.metadata["cd_min"])
        self.cd_max: float = float(self.metadata["cd_max"])

        arch = self.metadata["architecture"]
        self.model = PINNFRP(
            in_dim=len(self.feature_names),
            h1=int(arch["h1"]),
            h2=int(arch["h2"]),
            gate_h1=int(arch["gate_h1"]),
            gate_h2=int(arch["gate_h2"]),
            dropout=float(arch["dropout"]),
            use_failure_head=bool(arch["use_failure_head"]),
        ).to(self.device)

        try:
            ckpt = torch.load(MODEL_PATH, map_location=self.device, weights_only=True)
        except TypeError:
            ckpt = torch.load(MODEL_PATH, map_location=self.device)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.mu_x = torch.tensor(self.metadata["scalers"]["mu_x"], dtype=torch.float32, device=self.device)
        self.std_x = torch.tensor(self.metadata["scalers"]["std_x"], dtype=torch.float32, device=self.device)
        self.mu_y = torch.tensor(float(self.metadata["scalers"]["mu_y"]), dtype=torch.float32, device=self.device)
        self.std_y = torch.tensor(float(self.metadata["scalers"]["std_y"]), dtype=torch.float32, device=self.device)

    def predict_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        df = pd.DataFrame(records)
        return self.predict_dataframe(df).to_dict(orient="records")

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared = self._validate_inputs(df.copy())
        raw = torch.tensor(prepared[self.feature_names].to_numpy(), dtype=torch.float32, device=self.device)
        scaled = (raw - self.mu_x) / self.std_x

        with torch.no_grad():
            m_cr_s, m_fr_s, cd_cr_raw, cd_fr_raw, pfr_logits = self.model(scaled)
            p_fr = torch.sigmoid(pfr_logits).squeeze(1)
            z_hard = (p_fr >= self.threshold).float()

            if self.eval_gate_mode == "soft":
                gate = p_fr
            elif self.eval_gate_mode == "hard":
                gate = z_hard
            else:
                raise ValueError(f"Unsupported eval gate mode: {self.eval_gate_mode}")

            m_pred_s = (1.0 - gate.unsqueeze(1)) * m_cr_s + gate.unsqueeze(1) * m_fr_s
            m_over_bd = m_pred_s.squeeze(1) * self.std_y + self.mu_y

            b = raw[:, 0]
            d = raw[:, 1]
            fc = raw[:, 2]
            Af = raw[:, 3]
            Ef = raw[:, 4]
            ffu = raw[:, 5]

            phys = compute_aci_terms_from_raw_two_cd(
                b=b,
                d=d,
                fc=fc,
                Af=Af,
                Ef=Ef,
                ffu=ffu,
                cd_cr_raw=cd_cr_raw,
                cd_fr_raw=cd_fr_raw,
                cd_min=self.cd_min,
                cd_max=self.cd_max,
            )

            Mn_mix = (1.0 - gate) * phys.Mn_cr + gate * phys.Mn_fr
            T_sel = (1.0 - gate) * phys.T_cr + gate * phys.T_fr
            C_sel = (1.0 - gate) * phys.C_cr + gate * phys.C_fr
            eq_rel = torch.abs(T_sel - C_sel) / (0.5 * (torch.abs(T_sel) + torch.abs(C_sel)) + 1e-9)
            M_pred = m_over_bd * (b * d)

        out = prepared.copy()
        out["predicted_moment_kNm"] = M_pred.cpu().numpy()
        out["p_fr"] = p_fr.cpu().numpy()
        out["p_cc"] = (1.0 - p_fr).cpu().numpy()
        out["predicted_failure_mode"] = out["p_fr"].apply(lambda p: "FR" if p >= self.threshold else "CC")
        out["branch_moment_cr_kNm"] = phys.Mn_cr.cpu().numpy()
        out["branch_moment_fr_kNm"] = phys.Mn_fr.cpu().numpy()
        out["mixed_section_moment_kNm"] = Mn_mix.cpu().numpy()
        out["equilibrium_residual"] = eq_rel.cpu().numpy()
        out["c_over_d_cr"] = phys.cd_cr.cpu().numpy()
        out["c_over_d_fr"] = phys.cd_fr.cpu().numpy()
        out["eps_c_fr"] = phys.eps_c_fr.cpu().numpy()
        out["alpha_cr"] = phys.alpha_cr.cpu().numpy()
        out["beta_cr"] = phys.beta_cr.cpu().numpy()
        out["alpha_fr"] = phys.alpha_fr.cpu().numpy()
        out["beta_fr"] = phys.beta_fr.cpu().numpy()
        out["stress_ratio_cr_uncapped"] = phys.stress_ratio_cr_uncapped.cpu().numpy()
        return out

    def _validate_inputs(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [name for name in self.feature_names if name not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}. "
                f"Expected columns: {self.feature_names}"
            )

        df = df[self.feature_names].apply(pd.to_numeric, errors="coerce")
        if df.isna().any().any():
            raise ValueError("All inputs must be numeric and non-empty.")

        for column in self.feature_names:
            if (df[column] <= 0).any():
                raise ValueError(f"Column '{column}' must contain strictly positive values.")
        return df


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    return Predictor()
