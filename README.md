# Wildfire Risk Classifier

Classifies whether a house is **prone** to wildfire risk or **not prone**, from a photo of the house plus four attributes: state, roof material, wall material and ember-resistant vents.

A house counts as **prone** when its `fire_risk_percent` label is **50 or higher**. You can change this in `RISK_THRESHOLD` in `common.py`, then rerun all three steps.

## How to run

```bash
python split_data.py   # step 0: split the 100 labeled houses into 80 train / 20 test
python train.py        # step 1: train on the 80
python evaluate.py     # step 2: score on the 20 held-out houses
```

To classify a single new house:

```bash
python predict.py --image house.jpg --state california --roof tile --wall stucco --vents yes
```

Requirements: Python with `torch`, `torchvision`, `scikit-learn`, `pandas`, `matplotlib`, `joblib` and `pillow`.

## Folder layout

| Path | What it is |
|---|---|
| `split_data.py` | **Step 0.** Splits the 100 labeled houses 80/20 (fixed seed, same share of prone houses in each) |
| `train.py` | **Step 1.** Trains and picks the model, saves it to `models/` |
| `evaluate.py` | **Step 2.** Scores the saved model on the test houses, writes `results/` |
| `predict.py` | Optional. Classifies one new house |
| `common.py` | Shared settings (paths, threshold) and feature code. Not run directly |
| `data/labeled_100/images/` | The 100 labeled house photos, used by both train and test |
| `data/labeled_100/labels.csv` | Labels for all 100 (was `wildfire_date_100.csv`) |
| `data/train_labels.csv` | The 80 training houses, created by `split_data.py` |
| `data/test_labels.csv` | The 20 test houses, created by `split_data.py`. Never used in training or model selection |
| `models/wildfire_classifier.joblib` | The trained model, created by `train.py` |
| `results/cross_validation.csv` | Training-set comparison of the three candidate models |
| `results/test_metrics.txt` | Test accuracy, precision, recall and confusion matrix |
| `results/test_predictions.csv` | Per-house predictions on the test set |
| `results/confusion_matrix.png` | The confusion matrix as a picture |

## How the model works

1. **Image features.** Each photo goes through ResNet18, a CNN pretrained on ImageNet. It turns the photo into 512 numbers that describe what's in it. The CNN is frozen, meaning its pretrained weights aren't updated. Since the dataset only contains 80 training images, this helps reduce overfitting while still allowing us to extract useful visual features from each house.
2. **House attributes.** State, roof, wall and vents are one-hot encoded. County is left out because most counties appear only once or twice.
3. **Classifier.** A logistic regression predicts prone or not prone. `train.py` tries three versions: attributes only, image only, and both combined. It tunes each with 5-fold cross-validation repeated 10 times, **using only the 80 training houses**, and keeps the version with the best cross-validated accuracy.
4. **Test.** `evaluate.py` scores that one chosen model on the 20 held-out houses.

## Results

Run on 2026-11-12. Full output is in `results/`.

| Model | Cross-validated accuracy (80 training houses) | Test accuracy (20 held-out houses) |
|---|---|---|
| **tabular (chosen)** | **94.1%** (± 6.3%) | **100%** (20/20) |
| combined | 92.4% | 100% |
| image only | 80.9% | 85.0% |

Always guessing "prone" would score 55% on this test set.

**How to read this.** The realistic accuracy is **around 94%**, the cross-validated figure. With only 20 test houses, a perfect score is well within luck for a model that's right about 94% of the time. Treat 100% as "consistent with about 94%", not as a guarantee.

**What it learned.** Roof material, wall material and vents carry almost all the signal. Adding the photo doesn't help, and the photo alone is noticeably weaker (about 81%). Most likely the labels were assigned from those attributes, so the photo adds little that the attributes don't already say.

## Future Improvements

- **Expand the dataset.** Collect more labeled house images from different states and regions to improve generalization.
- **Improve image classification.** Fine-tune ResNet18 on a larger dataset to better recognize roof materials, vegetation and other visible wildfire risk factors.
- **Add environmental features.** Incorporate temperature, rainfall, vegetation density and proximity to forests for a more complete wildfire risk assessment.
- **Test on external data.** Evaluate the model on houses from a separate dataset to measure how well it performs on unseen properties.
- **Build a web interface or mobile app (currently being developed).** Allow users to upload a house photo, enter property details and receive a wildfire risk prediction.
