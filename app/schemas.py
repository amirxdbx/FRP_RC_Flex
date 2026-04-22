from __future__ import annotations

from pydantic import BaseModel, Field


class BeamInput(BaseModel):
    b_mm: float = Field(..., gt=0, description="Beam width in mm.")
    d_mm: float = Field(..., gt=0, description="Effective depth in mm.")
    fc_MPa: float = Field(..., gt=0, description="Concrete compressive strength in MPa.")
    Af_mm2: float = Field(..., gt=0, description="FRP reinforcement area in mm^2.")
    Ef_GPa: float = Field(..., gt=0, description="FRP elastic modulus in GPa.")
    ffu_MPa: float = Field(..., gt=0, description="FRP tensile strength in MPa.")


class PredictionOutput(BeamInput):
    predicted_moment_kNm: float
    p_fr: float
    p_cc: float
    predicted_failure_mode: str
    branch_moment_cr_kNm: float
    branch_moment_fr_kNm: float
    mixed_section_moment_kNm: float
    equilibrium_residual: float
    c_over_d_cr: float
    c_over_d_fr: float
    eps_c_fr: float
    alpha_cr: float
    beta_cr: float
    alpha_fr: float
    beta_fr: float
    stress_ratio_cr_uncapped: float


class BatchPredictionRequest(BaseModel):
    items: list[BeamInput]


class BatchPredictionResponse(BaseModel):
    items: list[PredictionOutput]

