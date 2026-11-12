"""OPTIONAL: classify one new house with the trained model.

    python predict.py --image path/to/house.jpg --state california --roof tile --wall stucco --vents yes

Run train.py first. Prints whether the house is prone to wildfire risk and the
model's probability. Attributes it never saw in training (e.g. a new state)
are allowed; they simply don't push the prediction either way.
"""

import argparse

import joblib
import pandas as pd

import common


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, help="photo of the house (.jpg)")
    parser.add_argument("--state", required=True, help="e.g. california")
    parser.add_argument("--roof", required=True, help="roof material, e.g. shingle, metal, tile, asphalt")
    parser.add_argument("--wall", required=True, help="wall material, e.g. wood, vinyl, stucco, brick, concrete")
    parser.add_argument("--vents", required=True, choices=["yes", "no"], help="has ember-resistant vents?")
    args = parser.parse_args()

    saved = joblib.load(common.MODEL_PATH)
    pipe = saved["pipelines"][saved["chosen"]]

    row = pd.DataFrame([{
        "state": args.state.strip().lower(),
        "roof_material": args.roof.strip().lower(),
        "wall_material": args.wall.strip().lower(),
        "safety_features_vents": args.vents,
    }])
    img = pd.DataFrame(common.image_features([args.image]), columns=common.IMAGE_COLUMNS)
    X = pd.concat([row, img], axis=1)

    prob = pipe.predict_proba(X)[0, 1]
    label = saved["class_names"][int(prob >= 0.5)]
    print(f"{label.upper()}  (probability prone: {prob:.0%}, model: {saved['chosen']})")


if __name__ == "__main__":
    main()
