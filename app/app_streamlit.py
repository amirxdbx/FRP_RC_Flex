from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.service import get_predictor


st.set_page_config(page_title="FRP-RC PINN Predictor", page_icon="📐", layout="wide")
st.title("FRP-RC PINN Predictor")
st.caption("Predict flexural capacity and CC/FR failure tendency using the trained physics-informed model.")

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
    width = 440
    height = 300
    margin_x = 56
    margin_y = 22
    section_w = 320
    section_h = 232
    x0 = margin_x
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
    <svg viewBox="0 0 {width} {height}" width="100%" xmlns="http://www.w3.org/2000/svg">
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
    </svg>
    """


predictor = load_predictor()

with st.sidebar:
    st.header("Required Inputs")
    st.markdown(
        "- `b_mm`: beam width in mm\n"
        "- `d_mm`: effective depth in mm\n"
        "- `fc_MPa`: concrete compressive strength in MPa\n"
        "- `Af_mm2`: FRP reinforcement area in mm²\n"
        "- or define reinforcement by bar count and bar diameter\n"
        "- `Ef_GPa`: FRP elastic modulus in GPa\n"
        "- `ffu_MPa`: FRP tensile strength in MPa"
    )

tab_single, tab_batch = st.tabs(["Single Prediction", "Batch CSV"])

with tab_single:
    col1, col2, col3 = st.columns(3)
    with col1:
        b_mm = st.number_input("b_mm", min_value=1.0, value=DEFAULT_INPUTS["b_mm"], step=1.0)
        d_mm = st.number_input("d_mm", min_value=1.0, value=DEFAULT_INPUTS["d_mm"], step=1.0)
    with col2:
        fc_MPa = st.number_input("fc_MPa", min_value=1.0, value=DEFAULT_INPUTS["fc_MPa"], step=0.1)
        af_mode = st.radio(
            "Reinforcement input",
            options=["Direct Af", "Bar count + diameter"],
            horizontal=False,
        )

        if af_mode == "Direct Af":
            Af_mm2 = st.number_input("Af_mm2", min_value=1.0, value=DEFAULT_INPUTS["Af_mm2"], step=1.0)
            viz_bars_direct = st.number_input(
                "Bars for graphic",
                min_value=1,
                max_value=12,
                value=DEFAULT_INPUTS["viz_bars_direct"],
                step=1,
                help="Used only to draw an equivalent section graphic when Af is entered directly.",
            )
            n_bars = int(viz_bars_direct)
            bar_diameter_mm = equivalent_bar_diameter(Af_mm2, n_bars)
            st.caption(
                f"Equivalent graphic only: {n_bars} bars of {bar_diameter_mm:.2f} mm gives Af ≈ {Af_mm2:.2f} mm²."
            )
        else:
            n_bars = int(
                st.number_input(
                    "Number of bars",
                    min_value=1,
                    max_value=12,
                    value=DEFAULT_INPUTS["n_bars"],
                    step=1,
                )
            )
            bar_diameter_mm = st.number_input(
                "Bar diameter (mm)",
                min_value=1.0,
                value=DEFAULT_INPUTS["bar_diameter_mm"],
                step=0.1,
            )
            Af_mm2 = area_from_bar_layout(n_bars, bar_diameter_mm)
            st.metric("Computed Af (mm²)", f"{Af_mm2:.2f}")
    with col3:
        Ef_GPa = st.number_input("Ef_GPa", min_value=1.0, value=DEFAULT_INPUTS["Ef_GPa"], step=0.1)
        ffu_MPa = st.number_input("ffu_MPa", min_value=1.0, value=DEFAULT_INPUTS["ffu_MPa"], step=1.0)

    preview_col, notes_col = st.columns([1.35, 1.0])
    with preview_col:
        components.html(
            render_section_svg(
                b_mm=b_mm,
                d_mm=d_mm,
                n_bars=n_bars,
                bar_diameter_mm=bar_diameter_mm,
                af_mm2=Af_mm2,
                mode_label=af_mode,
            ),
            height=320,
        )
    with notes_col:
        st.subheader("Section Summary")
        st.write(f"Beam width `b`: {b_mm:.1f} mm")
        st.write(f"Effective depth `d`: {d_mm:.1f} mm")
        st.write(f"Concrete strength `f_c`: {fc_MPa:.1f} MPa")
        st.write(f"FRP modulus `E_f`: {Ef_GPa:.1f} GPa")
        st.write(f"FRP tensile strength `f_fu`: {ffu_MPa:.1f} MPa")
        st.write(f"Reinforcement area `Af`: {Af_mm2:.2f} mm²")
        st.write(f"Reinforcement input: {af_mode}")
        st.write(f"Bars shown: {n_bars}")
        st.write(f"Bar diameter: {bar_diameter_mm:.2f} mm")

    if st.button("Predict", type="primary"):
        result = predictor.predict_records(
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

        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Predicted Moment (kN·m)", f"{result['predicted_moment_kNm']:.3f}")
        metric2.metric("Predicted Failure Mode", result["predicted_failure_mode"])
        metric3.metric("FR Probability", f"{result['p_fr']:.3f}")

        st.subheader("Detailed Outputs")
        st.dataframe(pd.DataFrame([result]), use_container_width=True)

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
