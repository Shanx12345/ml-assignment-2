# Shopper conversion model comparison

## a. Problem statement

Predict whether an e-commerce session ends in a purchase. `Revenue=1` indicates a purchase and `Revenue=0` indicates a session without a purchase. Since most sessions end without a purchase, accuracy alone can hide poor purchase detection.

## b. Dataset description

The source CSV contains 12,330 sessions, 17 predictors and the `Revenue` target. The experiment excludes `PageValues` because it is derived from transaction-related information and may be unavailable at prediction time. Removing 125 duplicate rows after that exclusion leaves 12,205 sessions, including 1,908 purchases and 10,297 non-purchases. The retained dataset meets the assignment minimum of 12 predictors and 500 observations.

| Feature type | Retained columns |
| --- | --- |
| Numeric (9) | `Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration`, `BounceRates`, `ExitRates`, `SpecialDay` |
| Categorical (7) | `Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType`, `Weekend` |

The source CSV has no missing cells. Integer category codes are one-hot encoded. The target is held separately from the 16 input features.

Dataset credit goes to C. Sakar and Y. Kastro (2018), [Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset), UCI Machine Learning Repository, [DOI 10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q), CC BY 4.0. The experiment excludes `PageValues`, removes duplicate retained rows, normalises labels and category representations, and partitions the data for training and evaluation.

## c. Repository and Application Links

- [GitHub repository](https://github.com/Shanx12345/ml-assignment-2)
- [Streamlit application](https://ml-assignment-2-lpjbf8fz3wh2xpkbxx8i9d.streamlit.app/)

## d. Models used

All five classifiers share the same stratified split, with 7,323 training, 2,441 validation and 2,441 test rows. The random seed is `20261107`. Three settings per model were compared on the validation set using MCC, with AUC as the tie-breaker. The selected settings were then refitted on the combined 9,764 training and validation rows. The test set was reserved for final evaluation.

Each pipeline fits its preprocessing alongside its classifier. It applies median imputation and standardisation to numeric features, with `log1p` before scaling for six count and duration features. It imputes categorical values with the most frequent category and then one-hot encodes them. Logistic Regression, Decision Tree and SVM use balanced class weights. KNN uses distance-weighted neighbours. Gaussian Naive Bayes supports the continuous, standardised inputs in this comparison.

### Test metric comparison

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.6678 | 0.7675 | 0.2840 | 0.7382 | 0.4102 | 0.2906 |
| Decision Tree | 0.7280 | 0.7449 | 0.3115 | 0.6099 | 0.4124 | 0.2836 |
| KNN | 0.8374 | 0.7258 | 0.3636 | 0.0524 | 0.0915 | 0.0866 |
| Gaussian Naive Bayes | 0.6112 | 0.7489 | 0.2608 | 0.8089 | 0.3944 | 0.2788 |
| Support Vector Machine | 0.6669 | 0.7800 | 0.2902 | 0.7801 | 0.4230 | 0.3136 |

Precision, recall and F1 refer to the purchase class. ROC AUC uses decision scores for Logistic Regression and SVM, and estimated purchase probabilities for the other models. The remaining metrics use the classifiers’ predicted labels. Full precision results are in [`results/test_metrics.csv`](results/test_metrics.csv).

### Observations on model performance

The test set contains 382 purchases and 2,059 non-purchases. Predicting no purchase for every session gives an accuracy of 0.8435 and detects zero purchases. All five classifiers fall below that accuracy baseline.

| Model | Held-out observations |
| --- | --- |
| Logistic Regression | Detected 282 of 382 purchases and flagged 711 non-purchases. |
| Decision Tree | Detected 233 purchases and flagged 515 non-purchases. Compared with Logistic Regression, it reduced false positives by 196 and detected 49 fewer purchases. |
| KNN | Had the highest accuracy among the five models but detected only 20 purchases. It missed 362, giving a recall of 0.0524. |
| Gaussian Naive Bayes | Detected the most purchases (309) and produced the most false positives (876). |
| Support Vector Machine | Detected 298 purchases and flagged 729 non-purchases. It had the highest test AUC (0.7800), F1 (0.4230) and MCC (0.3136). |

**Overall selection.** Logistic Regression was selected before test evaluation because it had the highest validation MCC (0.2865). SVM's held-out performance makes it a candidate for follow-up evaluation on a fresh split or new data.

### Files and execution

Use Python 3.12 and run these commands from the repository root.

```bash
python -m pip install -r requirements.txt
python train_models.py
python -m streamlit run app.py
```

Training saves the fitted pipelines in `model/` and the experiment outputs in `results/`. The app loads the saved pipelines and evaluates `test_data.csv` or a labelled upload containing the 16 predictors and `Revenue`. It displays model metrics, confusion matrices, classification reports and ROC curves, and offers prediction downloads. Uploading a CSV runs evaluation with the saved models.
