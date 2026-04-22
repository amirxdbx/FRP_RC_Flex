# FRP-RC PINN Predictor

Deployable inference app for the trained physics-informed neural network used in the manuscript. It supports:

- single prediction for one FRP-RC beam
- batch prediction from CSV
- a Streamlit UI for demos
- a FastAPI service for API-based deployment

## Inputs

The model expects these columns and units:

- `b_mm`: beam width in mm
- `d_mm`: effective depth in mm
- `fc_MPa`: concrete compressive strength in MPa
- `Af_mm2`: FRP reinforcement area in mm^2
- `Ef_GPa`: FRP elastic modulus in GPa
- `ffu_MPa`: FRP tensile strength in MPa

## Outputs

The app returns:

- predicted flexural capacity in kN·m
- FR probability `p_fr`
- CC probability `p_cc`
- predicted dominant failure mode
- branch moments and physics-related internal quantities

## Local Setup

```bash
pip install -r requirements.txt
```

## Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

## Run the FastAPI Service

```bash
uvicorn app.app_fastapi:app --reload
```

Swagger docs will be available at:

```text
http://127.0.0.1:8000/docs
```

## Run Batch Prediction from CLI

```bash
python -m app.predict_cli --input sample_input.csv --output predictions.csv
```

## Notes

- The bundled model artifact is `artifacts/best_pinn_phys.pth`.
- The reconstructed normalization statistics are stored in `artifacts/model_metadata.json`.
- If you later want a cleaner public-facing UI, keep FastAPI as the backend and put Streamlit or a custom frontend on top of it.
