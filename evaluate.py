"""STEP 2: score the trained model on the 20 held-out test houses.

    python evaluate.py

Run train.py first. Reads data/test_labels.csv and the photos in
data/labeled_100/images/, and writes:
  results/test_predictions.csv  one row per test house: true vs predicted
  results/test_metrics.txt      accuracy, precision, recall, confusion matrix
  results/confusion_matrix.png  the confusion matrix as a picture

The chosen model's score is the one to report. The other two candidates are
scored too, for reference only; they were not picked using these numbers.
"""

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix)

import common


def main():
    saved = joblib.load(common.MODEL_PATH)
    chosen = saved["chosen"]
    names = saved["class_names"]

    df = common.load_labels(common.TEST_LABELS)
    y = df["prone"].values
    print(f"Test houses: {len(df)}  (prone: {y.sum()}, not prone: {len(y) - y.sum()})")
    print("Extracting image features...")
    X = common.build_inputs(df, common.TEST_IMAGES)

    pipe = saved["pipelines"][chosen]
    pred = pipe.predict(X)
    prob = pipe.predict_proba(X)[:, 1]
    acc = accuracy_score(y, pred)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    report = classification_report(y, pred, labels=[0, 1], target_names=names, zero_division=0)

    others = []
    for name, other in saved["pipelines"].items():
        if name != chosen:
            others.append(f"  {name:<9} {accuracy_score(y, other.predict(X)):.1%}")

    text = "\n".join([
        f"Wildfire risk classifier: test results",
        f"Model: {chosen}   Threshold: fire_risk_percent >= {saved['risk_threshold']:g} means prone",
        f"Test houses: {len(df)}",
        "",
        f"ACCURACY: {acc:.1%}  ({(pred == y).sum()} of {len(y)} correct)",
        f"Always guessing one class would score: {max(y.mean(), 1 - y.mean()):.1%}",
        "",
        "Confusion matrix (rows = actual, columns = predicted):",
        f"                  pred not prone   pred prone",
        f"  not prone       {cm[0, 0]:>14}   {cm[0, 1]:>10}",
        f"  prone           {cm[1, 0]:>14}   {cm[1, 1]:>10}",
        "",
        report,
        "Other candidates on the same test set (reference only, not used to choose):",
        *others,
    ])
    print("\n" + text)

    common.RESULTS_DIR.mkdir(exist_ok=True)
    (common.RESULTS_DIR / "test_metrics.txt").write_text(text + "\n", encoding="utf-8")

    out = df[["filename", "state", "roof_material", "wall_material", "safety_features_vents",
              "fire_risk_percent"]].copy()
    out["actual"] = [names[v] for v in y]
    out["predicted"] = [names[v] for v in pred]
    out["prob_prone"] = prob.round(3)
    out["correct"] = pred == y
    out.to_csv(common.RESULTS_DIR / "test_predictions.csv", index=False)

    disp = ConfusionMatrixDisplay(cm, display_labels=names)
    disp.plot(cmap="Oranges", colorbar=False)
    plt.title(f"Test set: {acc:.0%} accuracy ({chosen} model)")
    plt.tight_layout()
    plt.savefig(common.RESULTS_DIR / "confusion_matrix.png", dpi=150)

    print("\nWrote results/test_metrics.txt, results/test_predictions.csv, results/confusion_matrix.png")


if __name__ == "__main__":
    main()
