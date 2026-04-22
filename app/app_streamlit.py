from __future__ import annotations

import pandas as pd
import streamlit as st

from app.service import get_predictor


st.set_page_config(page_title="FRP-RC PINN Predictor", page_icon="📐", layout="wide")
st.title("FRP-RC PINN Predictor")
st.caption("Predict flexural capacity and CC/FR failure tendency using the trained physics-informed model.")


@st.cache_resource
def load_predictor():
    return get_predictor()


predictor = load_predictor()

with st.sidebar:
    st.header("Required Inputs")
    st.markdown(
        "- `b_mm`: beam width in mm\n"
        "- `d_mm`: effective depth in mm\n"
        "- `fc_MPa`: concrete compressive strength in MPa\n"
        "- `Af_mm2`: FRP reinforcement area in mm²\n"
        "- `Ef_GPa`: FRP elastic modulus in GPa\n"
        "- `ffu_MPa`: FRP tensile strength in MPa"
    )

tab_single, tab_batch = st.tabs(["Single Prediction", "Batch CSV"])

with tab_single:
    col1, col2, col3 = st.columns(3)
    with col1:
        b_mm = st.number_input("b_mm", min_value=1.0, value=189.4, step=1.0)
        d_mm = st.number_input("d_mm", min_value=1.0, value=265.9, step=1.0)
    with col2:
        fc_MPa = st.number_input("fc_MPa", min_value=1.0, value=43.27, step=0.1)
        Af_mm2 = st.number_input("Af_mm2", min_value=1.0, value=449.48, step=1.0)
    with col3:
        Ef_GPa = st.number_input("Ef_GPa", min_value=1.0, value=57.98, step=0.1)
        ffu_MPa = st.number_input("ffu_MPa", min_value=1.0, value=1015.47, step=1.0)

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
    st.write("Upload a CSV with columns: `b_mm,d_mm,fc_MPa,Af_mm2,Ef_GPa,ffu_MPa`.")
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
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

