"""STEP 1: train the wildfire risk classifier on the 80 training houses.

    python train.py

Run split_data.py first. What it does:
  1. Loads data/train_labels.csv and the photos in data/labeled_100/images/.
  2. Turns each photo into 512 numbers with a pretrained ResNet18.
  3. Compares three candidate models using cross-validation on the
     training set only (the test set is never touched here):
        - tabular:  house attributes only (state, roof, walls, vents)
        - image:    photo features only
        - combined: both
  4. Picks the candidate with the best cross-validated accuracy, retrains
     it on all 80 training houses, and saves it to models/wildfire_classifier.joblib.
  5. Writes the comparison to results/cross_validation.csv.

Then run evaluate.py to score the saved model on the 20 test houses.
"""

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import common


def make_candidates():
    """Each candidate: (preprocessing for its columns, hyperparameters to search)."""
    onehot = OneHotEncoder(handle_unknown="ignore")  # unseen states etc. become all-zeros
    image_prep = Pipeline([("scale", StandardScaler()), ("pca", PCA(random_state=common.RANDOM_SEED))])
    C_values = [0.001, 0.01, 0.1, 1, 10]
    pca_sizes = [5, 10, 20, 40]

    return {
        "tabular": (
            ColumnTransformer([("tab", onehot, common.TABULAR_COLUMNS)]),
            {"clf__C": C_values},
        ),
        "image": (
            ColumnTransformer([("img", image_prep, common.IMAGE_COLUMNS)]),
            {"clf__C": C_values, "prep__img__pca__n_components": pca_sizes},
        ),
        "combined": (
            ColumnTransformer([
                ("tab", onehot, common.TABULAR_COLUMNS),
                ("img", image_prep, common.IMAGE_COLUMNS),
            ]),
            {"clf__C": C_values, "prep__img__pca__n_components": pca_sizes},
        ),
    }


def main():
    df = common.load_labels(common.TRAIN_LABELS)
    y = df["prone"].values
    print(f"Training houses: {len(df)}  "
          f"(prone: {y.sum()}, not prone: {len(y) - y.sum()}, "
          f"threshold: fire_risk_percent >= {common.RISK_THRESHOLD:g})")

    print("Extracting image features with pretrained ResNet18...")
    X = common.build_inputs(df, common.TRAIN_IMAGES)

    # 5-fold cross-validation repeated 10 times with different shuffles,
    # so the score doesn't hinge on one lucky or unlucky split.
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=common.RANDOM_SEED)

    rows, searches = [], {}
    for name, (prep, grid) in make_candidates().items():
        pipe = Pipeline([
            ("prep", prep),
            ("clf", LogisticRegression(max_iter=5000, random_state=common.RANDOM_SEED)),
        ])
        search = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy", n_jobs=1)
        search.fit(X, y)
        i = search.best_index_
        mean = search.cv_results_["mean_test_score"][i]
        std = search.cv_results_["std_test_score"][i]
        rows.append({"model": name, "cv_accuracy": round(mean, 3), "cv_std": round(std, 3),
                     "best_settings": search.best_params_})
        searches[name] = search
        print(f"  {name:<9} cross-validated accuracy: {mean:.1%} (+/- {std:.1%})  {search.best_params_}")

    cv_table = pd.DataFrame(rows).sort_values("cv_accuracy", ascending=False)
    common.RESULTS_DIR.mkdir(exist_ok=True)
    cv_table.to_csv(common.RESULTS_DIR / "cross_validation.csv", index=False)

    best = cv_table.iloc[0]["model"]
    print(f"\nChosen model: {best}  (best cross-validated accuracy)")

    common.MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({
        "chosen": best,
        "pipelines": {name: s.best_estimator_ for name, s in searches.items()},
        "risk_threshold": common.RISK_THRESHOLD,
        "class_names": common.CLASS_NAMES,
    }, common.MODEL_PATH)
    print(f"Saved to {common.MODEL_PATH.relative_to(common.ROOT)}")
    print("Next: python evaluate.py")


if __name__ == "__main__":
    main()
