
# Heart Disease Prediction

Binary classification of heart disease using the [UCI Heart Disease (Cleveland) dataset](https://archive.ics.uci.edu/dataset/45/heart+disease). The project compares a Logistic Regression baseline with Random Forest and XGBoost models, prioritizing **recall**, since missing a patient with heart disease is more costly than a false alarm.

## Project Structure

```
Heart-Disease-Prediction/
├── data/
│   ├── heart_disease_data.csv    # Raw data fetched from the UCI repository
│   └── processed_data.csv        # Cleaned, encoded and scaled data used for training
├── models/
│   ├── logistic.joblib           # Tuned Logistic Regression
│   ├── randomforest.joblib       # Tuned Random Forest
│   └── xgboost.joblib            # XGBoost tuned with Bayesian optimization
├── eda.ipynb                     # Data cleaning, exploration and preprocessing
├── models.ipynb                  # Model training, tuning and evaluation
├── main.py
├── pyproject.toml
└── uv.lock
```

## Setup

The project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone <repo-url>
cd Heart-Disease-Prediction
uv sync
uv run jupyter lab
```

Run the notebooks in order: `eda.ipynb` first (it downloads the data and creates `data/processed_data.csv`), then `models.ipynb`. Make sure the `data/` and `models/` folders exist before running, as the notebooks save files into them.

## Dataset

The Cleveland dataset contains 303 patients and 13 clinical features, including age, sex, chest pain type, resting blood pressure, cholesterol, maximum heart rate, exercise-induced angina, ST depression (`oldpeak`), number of major vessels (`ca`) and thalassemia (`thal`). The original target `num` ranges from 0 (no disease) to 4 and was converted to binary: 0 = no disease, 1 = disease present.

## Exploratory Data Analysis (`eda.ipynb`)

- **Missing values:** Only 6 values were missing (4 in `ca`, 2 in `thal`). Formal tests of the missingness mechanism (e.g. Little's MCAR test) are not informative with so few missing values, so the affected rows were dropped.
- **Duplicates:** None found.
- **Outliers:** Detected with the 1.5 × IQR rule on the continuous features (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`) and removed.
- **Final dataset:** 278 patients, 154 without disease (55%) and 124 with disease (45%), so the classes are reasonably balanced.
- **Exploration:** Distributions of the target, age, sex and chest pain type, plus a correlation heatmap.
- **Preprocessing:** One-hot encoding for nominal features (`cp`, `restecg`, `slope`, `thal`), standardization for numeric features (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`, `ca`), and binary features (`sex`, `fbs`, `exang`) left unchanged.

## Modelling (`models.ipynb`)

The data is split 70/30 into training (194 patients) and test (84 patients) sets, stratified by the target. All models are tuned using the **F2 score** with 5-fold cross-validation on the training set only, so the test set is used once for final evaluation.

| Model                          | Tuning method                                                                                                  |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| Logistic Regression (baseline) | Grid search over regularization strength`C`                                                                  |
| Random Forest                  | Grid search over number of trees, max features, leaf nodes, split and leaf sizes, bootstrap                    |
| XGBoost                        | Bayesian optimization with Hyperopt (TPE, 100 trials) over depth, learning rate, subsample and number of trees |

## Results

Test set performance (84 patients, 37 with heart disease):

| Model               | Accuracy        | Recall          | Precision       | ROC-AUC        |
| ------------------- | --------------- | --------------- | --------------- | -------------- |
| Logistic Regression | 0.869           | **0.811** | 0.882           | **0.95** |
| Random Forest       | 0.845           | 0.757           | 0.875           | 0.92           |
| XGBoost             | **0.881** | **0.811** | **0.909** | 0.93           |

Logistic Regression and XGBoost perform almost identically, each identifying 30 of the 37 patients with heart disease. **Logistic Regression is the preferred model**: it matches XGBoost on recall, has the highest ROC-AUC, and is far more interpretable. Its strong performance suggests the relationship between the features and heart disease is largely additive, and the small training set limits the benefit of more complex models.

See the Analysis section of `models.ipynb` for the full discussion.

## Using a Saved Model

```python
from joblib import load

model = load("models/logistic.joblib")
predictions = model.predict(X)  # X must be preprocessed the same way as processed_data.csv
```

The preprocessing transformer is not saved, so new raw data must be encoded and scaled with the same steps as in `eda.ipynb` before prediction.

## Limitations

- The test set is small (84 patients), so differences of one or two patients between models are within noise.
- Continuous features were standardized before the train/test split, so a small amount of test-set information influenced preprocessing.
- Outlier removal may exclude clinically valid extreme values.

## Future Work

- Move preprocessing into a scikit-learn `Pipeline` fitted on training data only, and save it with the model
- Compare models with repeated stratified cross-validation
- Tune the decision threshold to increase recall
- Add model interpretation (e.g. Logistic Regression coefficients or SHAP values)

## Acknowledgements

Data: Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X
