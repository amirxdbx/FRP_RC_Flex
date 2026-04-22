from __future__ import annotations

import argparse

import pandas as pd

from app.service import get_predictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch inference for the FRP-RC PINN predictor.")
    parser.add_argument("--input", required=True, help="Path to input CSV.")
    parser.add_argument("--output", default="predictions.csv", help="Path to output CSV.")
    args = parser.parse_args()

    predictor = get_predictor()
    df = pd.read_csv(args.input)
    result = predictor.predict_dataframe(df)
    result.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()

