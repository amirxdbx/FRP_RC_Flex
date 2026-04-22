from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.service import get_predictor


st.set_page_config(page_title="FRP-RC PINN Predictor", page_icon="📐", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .hero-card {
        padding: 1.1rem 1.2rem 0.95rem 1.2rem;
        border: 1px solid rgba(11, 110, 79, 0.14);
        border-radius: 18px;
        background: linear-gradient(135deg, #f4fbf7 0%, #ffffff 55%, #eef6f2 100%);
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        color: #0B6E4F;
        margin: 0 0 0.2rem 0;
        line-height: 1.1;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #40534c;
        margin: 0;
    }
    .section-label {
        font-size: 0.9rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #0B6E4F;
        margin-top: 0.2rem;
        margin-bottom: 0.15rem;
    }
    .section-note {
        font-size: 0.92rem;
        color: #5b6761;
        margin-top: 0;
        margin-bottom: 0.9rem;
    }
    .summary-card {
        padding: 0.9rem 1rem;
        border-radius: 16px;
        background: #f7faf8;
        border: 1px solid rgba(17, 17, 17, 0.08);
    }
    .section-figure-shell {
        width: 100%;
        padding: 0.35rem;
        border-radius: 18px;
        background: #f7faf8;
        border: 1px solid rgba(17, 17, 17, 0.06);
        line-height: 0;
    }
    .section-figure-shell svg {
        display: block;
        width: 100%;
        height: auto;
    }
    </style>
    <div class="hero-card">
      <div class="hero-title">FRP-RC PINN Predictor</div>
      <p class="hero-subtitle">
        Predict flexural capacity and CC/FR failure tendency using the trained physics-informed model.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

DEFAULT_INPUTS = {
    "b_mm": 200.0,
    "d_mm": 254.0,
    "fc_MPa": 39.6,
    "Af_mm2": 393.0,
    "n_bars": 4,
    "bar_diameter_mm": 11.2,
    "viz_bars_direct": 4,
    "Ef_GPa": 48.7,
    "ffu_MPa": 995.0,
}


@st.cache_resource
def load_predictor():
    return get_predictor()


def area_from_bar_layout(n_bars: int, bar_diameter_mm: float) -> float:
    return n_bars * math.pi * (bar_diameter_mm ** 2) / 4.0


def equivalent_bar_diameter(af_mm2: float, n_bars: int) -> float:
    return math.sqrt((4.0 * af_mm2) / (max(n_bars, 1) * math.pi))


def render_section_svg(
    b_mm: float,
    d_mm: float,
    n_bars: int,
    bar_diameter_mm: float,
    af_mm2: float,
    mode_label: str,
) -> str:
    width = 550
    height = 375
    margin_x = 70
    margin_y = 28
    max_section_w = 400
    max_section_h = 290

    # Scale the sketch to the actual b:d proportion instead of forcing
    # every section into the same rectangle.
    scale = min(max_section_w / b_mm, max_section_h / d_mm)
    section_w = b_mm * scale
    section_h = d_mm * scale

    x0 = margin_x + (max_section_w - section_w) / 2
    y0 = margin_y
    cover = 24
    stirrup_inset = 18
    y_bars = y0 + section_h - cover - 10

    max_bar_px = 16
    scaled_bar_r = max(5, min(max_bar_px, 0.45 * bar_diameter_mm))

    if n_bars == 1:
        centers = [x0 + section_w / 2]
    else:
        left = x0 + cover + scaled_bar_r + 6
        right = x0 + section_w - cover - scaled_bar_r - 6
        spacing = (right - left) / (n_bars - 1)
        centers = [left + i * spacing for i in range(n_bars)]

    circles = "\n".join(
        f"<circle cx='{cx:.1f}' cy='{y_bars:.1f}' r='{scaled_bar_r:.1f}' "
        "fill='#0D7A57' stroke='#064E3B' stroke-width='2.2' />"
        for cx in centers
    )

    return f"""
    <svg viewBox="0 0 {width} {height}" preserveAspectRatio="xMidYMid meet"
         xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="concreteGrad" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="#ECE8DF"/>
          <stop offset="100%" stop-color="#D3CEC4"/>
        </linearGradient>
        <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
          <feDropShadow dx="0" dy="6" stdDeviation="6" flood-color="#000000" flood-opacity="0.16"/>
        </filter>
        <pattern id="speckle" width="28" height="28" patternUnits="userSpaceOnUse">
          <circle cx="6" cy="8" r="1.3" fill="#BBB3A7"/>
          <circle cx="18" cy="10" r="1.1" fill="#B2AA9D"/>
          <circle cx="11" cy="20" r="1.4" fill="#C2BAAF"/>
          <circle cx="22" cy="22" r="1.0" fill="#B7B0A5"/>
        </pattern>
      </defs>

      <rect x="0" y="0" width="{width}" height="{height}" fill="#F7FAF8" rx="18"/>
      <rect x="{x0}" y="{y0}" width="{section_w}" height="{section_h}" rx="14"
            fill="url(#concreteGrad)" stroke="#5F5A52" stroke-width="3.2" filter="url(#shadow)"/>
      <rect x="{x0}" y="{y0}" width="{section_w}" height="{section_h}" rx="14"
            fill="url(#speckle)" opacity="0.55"/>
      <rect x="{x0 + stirrup_inset}" y="{y0 + stirrup_inset}"
            width="{section_w - 2 * stirrup_inset}" height="{section_h - 2 * stirrup_inset}"
            rx="10" fill="none" stroke="#8F877A" stroke-width="4"/>
      <rect x="{x0 + 10}" y="{y0 + 10}" width="{section_w - 20}" height="{section_h * 0.26:.1f}"
            rx="10" fill="#FFFFFF" opacity="0.16"/>
      {circles}
      <line x1="{x0 + cover}" y1="{y_bars}" x2="{x0 + section_w - cover}" y2="{y_bars}"
            stroke="#7C7468" stroke-width="2" stroke-dasharray="5 5" opacity="0.55"/>
      <text x="{x0 + section_w / 2:.1f}" y="{y0 + section_h + 28:.1f}"
            text-anchor="middle" font-size="13" fill="#666666">
        scaled to b:d = {b_mm:.0f}:{d_mm:.0f}
      </text>
    </svg>
    """


predictor = load_predictor()

tab_single, tab_batch = st.tabs(["Single Prediction", "Batch CSV"])

with tab_single:
    input_col, preview_shell = st.columns([1, 0.65], gap="large")

    with input_col:
        st.markdown('<div class="section-label">Geometry</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-note">Define the beam cross-section used by the predictor.</p>',
            unsafe_allow_html=True,
        )
        st.latex(r"\text{Geometry:}\quad b,\ d")
        g1, g2 = st.columns(2)
        with g1:
            b_mm = st.number_input(
                r"Beam width, $b$ (mm)",
                min_value=1.0,
                value=DEFAULT_INPUTS["b_mm"],
                step=1.0,
                help="Section width measured in millimetres.",
            )
        with g2:
            d_mm = st.number_input(
                r"Effective depth, $d$ (mm)",
                min_value=1.0,
                value=DEFAULT_INPUTS["d_mm"],
                step=1.0,
                help="Distance from compression face to FRP reinforcement centroid.",
            )

        st.markdown('<div class="section-label">Reinforcement</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-note">Provide FRP area directly or derive it from bar count and bar diameter.</p>',
            unsafe_allow_html=True,
        )
        st.latex(r"A_f = n \cdot \frac{\pi \phi^2}{4}")
        af_mode = st.radio(
            "Reinforcement input mode",
            options=["Direct Af", "Bar count + diameter"],
            horizontal=True,
            label_visibility="collapsed",
        )

        if af_mode == "Direct Af":
            r1, r2 = st.columns([1.25, 1.0])
            with r1:
                Af_mm2 = st.number_input(
                    r"FRP area, $A_f$ (mm²)",
                    min_value=1.0,
                    value=DEFAULT_INPUTS["Af_mm2"],
                    step=1.0,
                )
            with r2:
                viz_bars_direct = st.number_input(
                    "Bars shown in preview",
                    min_value=1,
                    max_value=12,
                    value=DEFAULT_INPUTS["viz_bars_direct"],
                    step=1,
                    help="Only used to draw an equivalent layout in the section preview.",
                )
            n_bars = int(viz_bars_direct)
            bar_diameter_mm = equivalent_bar_diameter(Af_mm2, n_bars)
            st.caption(
                f"Equivalent preview layout: {n_bars} bars with diameter {bar_diameter_mm:.2f} mm."
            )
        else:
            r1, r2, r3 = st.columns([0.9, 1.1, 1.0])
            with r1:
                n_bars = int(
                    st.number_input(
                        r"Bar count, $n$",
                        min_value=1,
                        max_value=12,
                        value=DEFAULT_INPUTS["n_bars"],
                        step=1,
                    )
                )
            with r2:
                bar_diameter_mm = st.number_input(
                    r"Bar diameter, $\phi$ (mm)",
                    min_value=1.0,
                    value=DEFAULT_INPUTS["bar_diameter_mm"],
                    step=0.1,
                )
            Af_mm2 = area_from_bar_layout(n_bars, bar_diameter_mm)
            with r3:
                st.metric(r"Computed $A_f$", f"{Af_mm2:.2f} mm²")

        st.markdown('<div class="section-label">Material Properties</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-note">Specify the concrete and FRP mechanical properties.</p>',
            unsafe_allow_html=True,
        )
        st.latex(r"\text{Material inputs:}\quad f'_c,\ E_f,\ f_{fu}")
        m1, m2, m3 = st.columns(3)
        with m1:
            fc_MPa = st.number_input(
                r"Concrete strength, $f'_c$ (MPa)",
                min_value=1.0,
                value=DEFAULT_INPUTS["fc_MPa"],
                step=0.1,
            )
        with m2:
            Ef_GPa = st.number_input(
                r"FRP modulus, $E_f$ (GPa)",
                min_value=1.0,
                value=DEFAULT_INPUTS["Ef_GPa"],
                step=0.1,
            )
        with m3:
            ffu_MPa = st.number_input(
                r"FRP tensile strength, $f_{fu}$ (MPa)",
                min_value=1.0,
                value=DEFAULT_INPUTS["ffu_MPa"],
                step=1.0,
            )

        submitted = st.button("Predict Capacity", type="primary", use_container_width=True)

    if submitted:
        st.session_state["single_prediction_result"] = predictor.predict_records(
            [
                {
                    "b_mm": b_mm,
                    "d_mm": d_mm,
                    "fc_MPa": fc_MPa,
                    "Af_mm2": Af_mm2,
                    "Ef_GPa": Ef_GPa,
                    "ffu_MPa": ffu_MPa,
                }
            ]
        )[0]

    with preview_shell:
        st.markdown(
            (
                '<div class="section-figure-shell">'
                + render_section_svg(
                    b_mm=b_mm,
                    d_mm=d_mm,
                    n_bars=n_bars,
                    bar_diameter_mm=bar_diameter_mm,
                    af_mm2=Af_mm2,
                    mode_label=af_mode,
                )
                + "</div>"
            ),
            unsafe_allow_html=True,
        )
        result = st.session_state.get("single_prediction_result")
        if result is not None:
            st.markdown("---")
            st.markdown('<div class="section-label">Prediction Output</div>', unsafe_allow_html=True)
            st.latex(r"\hat{M},\ p_{\mathrm{FR}},\ p_{\mathrm{CC}}")
            out1, out2, out3 = st.columns(3)
            out1.metric(r"Predicted $\hat{M}$", f"{result['predicted_moment_kNm']:.1f} kN·m")
            out2.metric("Failure Mode", result["predicted_failure_mode"])
            out3.metric(r"$p_{\mathrm{FR}}$", f"{result['p_fr']:.3f}")

with tab_batch:
    st.write(
        "Upload a CSV with columns `b_mm,d_mm,fc_MPa,Af_mm2,Ef_GPa,ffu_MPa` "
        "or with `n_bars` and `bar_diameter_mm`; the app will compute `Af_mm2` automatically."
    )
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        if {"n_bars", "bar_diameter_mm"}.issubset(df.columns) and "Af_mm2" not in df.columns:
            df["Af_mm2"] = area_from_bar_layout(df["n_bars"], df["bar_diameter_mm"])
        st.subheader("Input Preview")
        st.dataframe(df.head(), use_container_width=True)

        if st.button("Run Batch Prediction"):
            result_df = predictor.predict_dataframe(df)
            st.subheader("Predictions")
            st.dataframe(result_df, use_container_width=True)
            st.download_button(
                label="Download predictions.csv",
                data=result_df.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )
