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
streamlit run app/app_streamlit.py
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

## Recommended Deployment Options

### Option 1: Streamlit Community Cloud

Best for a quick interactive demo.

Entry point:

```text
app/app_streamlit.py
```

### Option 2: Render or Railway with FastAPI

Better if you want a stable API that can later be consumed by Streamlit, a custom frontend, or another service.

This repo already includes `render.yaml` for Render deployment.

## GitHub Push

After creating a GitHub repository, run:

```bash
git init
git add .
git commit -m "Add FRP-RC PINN predictor app"
git branch -M main
git remote add origin <YOUR_REPO_URL>
git push -u origin main
```

If you want me to push it for you from this machine, send the repository URL and make sure GitHub authentication is available here.
