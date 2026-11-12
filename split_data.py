"""STEP 0: split the 100 labeled houses into a training set and a test set.

    python split_data.py

Reads data/labeled_100/labels.csv and writes:
  data/train_labels.csv   80 houses the model learns from
  data/test_labels.csv    20 houses held back to score the model

Both files point at the same photos in data/labeled_100/images/; no images are
copied or moved. The split keeps the same share of "prone" houses in each part
and uses a fixed random seed, so rerunning this gives the same split.
"""

from sklearn.model_selection import train_test_split

import common

TEST_SIZE = 20


def main():
    df = common.load_labels(common.LABELED_CSV)
    raw = df.drop(columns="prone")  # write the CSV columns unchanged
    train, test = train_test_split(raw, test_size=TEST_SIZE, stratify=df["prone"],
                                   random_state=common.RANDOM_SEED)
    train.sort_values("filename").to_csv(common.TRAIN_LABELS, index=False)
    test.sort_values("filename").to_csv(common.TEST_LABELS, index=False)

    for name, part in [("train", train), ("test", test)]:
        prone = int((part["fire_risk_percent"] >= common.RISK_THRESHOLD).sum())
        print(f"{name}: {len(part)} houses (prone: {prone}, not prone: {len(part) - prone})")
    print("Wrote data/train_labels.csv and data/test_labels.csv. Next: python train.py")


if __name__ == "__main__":
    main()
