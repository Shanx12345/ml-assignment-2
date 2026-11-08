# Shopper conversion model comparison

## a. Problem statement

Classify whether an e-commerce session ends in a purchase, using observed session attributes. Revenue=1 means a purchase; Revenue=0 means no purchase. The practical question is whether the models identify purchasers rather than simply predict the larger non-purchase class. This is retrospective session classification, not a validated live targeting system.

## b. Dataset description

Source: [UCI Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset).

The downloaded CSV has 12,330 rows, 17 predictors and one target. PageValues is excluded because its transaction-related definition raises an availability and leakage concern. That leaves 16 predictors: nine numeric and seven categorical. There are no missing cells in the source CSV. Removing 125 identical retained rows leaves 12,205 rows, with 1,908 purchases and 10,297 non-purchases. These counts were computed from the downloaded file.

Numeric predictors: Administrative, Administrative_Duration, Informational, Informational_Duration, ProductRelated, ProductRelated_Duration, BounceRates, ExitRates, SpecialDay.

Categorical predictors: Month, OperatingSystems, Browser, Region, TrafficType, VisitorType, Weekend. Integer category codes are encoded as categories, not treated as magnitudes. The target is excluded from the input features.

The dataset meets both minimums: 16 retained predictors exceeds 12, and 12,205 retained rows exceeds 500. UCI dataset credit: Sakar, C. and Kastro, Y. (2018), DOI [10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q), CC BY 4.0. Data modifications: PageValues excluded, duplicate retained rows removed, target and category representations normalised, and train/validation/test partitions created.

## c. Github Repository Link

https://github.com/Shanx12345/ml-assignment-2

Streamlit App: https://ml-assignment-2-lpjbf8fz3wh2xpkbxx8i9d.streamlit.app/

## d. Models used

All five models use the same retained dataset and identical partitions. The split is stratified: 7,323 training rows, 2,441 validation rows and 2,441 held-out test rows. Random seed: 20261107. Each family has three candidate settings; validation MCC selects the setting, with validation AUC as tie-breaker. The overall candidate is also selected before test evaluation. Each selected pipeline is then refitted on the 9,764 training plus validation rows and evaluated once on the same held-out test set.

Preprocessing is fitted inside each pipeline: median imputation, log1p for six non-negative count/duration columns, numerical standardisation, most-frequent categorical imputation and one-hot encoding. Pipelines contain both the transformations and the classifier. The app loads these saved pipelines and never trains on an uploaded CSV.

Logistic Regression, Decision Tree and SVM use balanced class weights. KNN and Gaussian Naive Bayes use their standard unweighted fitting. These are comparisons of documented configured pipelines, not a claim that each model family has been exhaustively optimised. Gaussian Naive Bayes is used because Multinomial Naive Bayes cannot accept the negative values produced by standardisation.

### Test metric comparison

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.6678 | 0.7675 | 0.2840 | 0.7382 | 0.4102 | 0.2906 |
| Decision Tree | 0.7280 | 0.7449 | 0.3115 | 0.6099 | 0.4124 | 0.2836 |
| KNN | 0.8374 | 0.7258 | 0.3636 | 0.0524 | 0.0915 | 0.0866 |
| Naive Bayes | 0.6112 | 0.7489 | 0.2608 | 0.8089 | 0.3944 | 0.2788 |
| Support Vector Machine | 0.6669 | 0.7800 | 0.2902 | 0.7801 | 0.4230 | 0.3136 |

All numbers are measured results from the bundled reference run, rounded to four decimals for display. Precision, recall and F1 use the purchase class, not macro or weighted averages. AUC uses continuous decision scores for Logistic Regression and SVM, and purchase probabilities for the other models. The other five metrics use native predict outputs. The test set has 382 purchases and 2,059 non-purchases. An always-no-purchase baseline has accuracy 0.8435, purchase precision/recall/F1/MCC of zero by the stated zero-division convention, and AUC 0.5000 for a constant score.

### Observations on model performance

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Linear reference with class balancing. Its validation selection status is reported in the overall decision below. Purchases detected: 282/382; false positives: 711; precision 0.2840; recall 0.7382. |
| Decision Tree | A shallow nonlinear rule set, limited by maximum depth and minimum leaf size. The error counts below show its precision and recall trade-off. Purchases detected: 233/382; false positives: 515; precision 0.3115; recall 0.6099. |
| KNN | Distance-weighted neighbours favour the majority class in this run. Its relatively high accuracy hides weak purchase detection. Purchases detected: 20/382; false positives: 35; precision 0.3636; recall 0.0524. |
| Naive Bayes | Correlated predictors and non-Gaussian indicator features limit its assumptions. Read purchase detection alongside the false-positive count. Purchases detected: 309/382; false positives: 876; precision 0.2608; recall 0.8089. |
| Support Vector Machine | A balanced nonlinear margin classifier using the RBF kernel. Test scores describe its performance; the initial selection uses validation results. Purchases detected: 298/382; false positives: 729; precision 0.2902; recall 0.7801. |
| Overall winner for this experiment | Logistic Regression was selected before test evaluation, using validation MCC and then AUC. Support Vector Machine has the strongest held-out MCC in this particular run. A fresh validation or prospective experiment is needed before changing the selected model. |

No model exceeds the majority baseline on accuracy in this experiment. Several detect substantially more purchases; whether that is useful depends on the cost of false positives and missed purchases. No monetary benefit or causal lift is established. The models are not calibrated probability estimators, and the SVM output is a margin. The ranking can change with feature availability, split, class weights, preprocessing and decision threshold. Only a small parameter search and one random split were used.

### Files and execution

The repository includes app.py, requirements.txt, README.md, test_data.csv, five .joblib pipelines in model/, training source, shared preprocessing source, source data, split files, metrics, predictions, confusion matrices and a verification script. The Streamlit app displays upload, selection, all-model metrics, confusion matrix, classification report, ROC comparison and prediction downloads.


### Reproducibility and limitations

Execution recorded in current artifacts: Assistant workspace reference run; not BITS Virtual Lab evidence.
Source SHA-256: b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9.
Versions: Python 3.12.14, scikit-learn 1.8.0, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, joblib 1.5.3.

Install the pinned requirements and use Python 3.12 for the closest reproduction. Retraining creates new model artifacts; changing seeds, preprocessing or package versions may change the results. Never edit metric values to match the reference. Run readme_tools.py after retraining so the tables reflect the current artifacts. If you change the experiment design, rewrite the descriptive text and observations to match it.

Features represent observed sessions. Excluding PageValues alone does not establish that all other features were available early enough for a live decision. The dataset has no usable per-row timestamp for a temporal deployment test. This random split therefore estimates performance on similar sampled sessions, not future conversion lift.
