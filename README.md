# Heart Disease Prediction

Binary classification of heart disease using the [UCI Heart Disease (Cleveland) dataset](https://archive.ics.uci.edu/dataset/45/heart+disease). The project compares a Logistic Regression baseline with Random Forest and XGBoost models, prioritizing **recall**, since missing a patient with heart disease is more costly than a false alarm.

🔗 **[Live demo](https://heart-disease-prediction-wwega8j3hbfpwrk8wd6xhq.streamlit.app/)**

## Project Structure

```
Heart-Disease-Prediction/
├── data/
│   ├── heart_disease_data.csv    # Raw data fetched from the UCI repository
│   └── processed_data.csv        # Cleaned, encoded and scaled data used for training
├── models/
│   ├── preprocessor.joblib       # Fitted encoder and scaler used by the app
│   ├── logistic.joblib           # Tuned Logistic Regression
│   ├── randomforest.joblib       # Tuned Random Forest
│   └── xgboost.joblib            # XGBoost tuned with Bayesian optimization
├── eda.ipynb                     # Data cleaning, exploration and preprocessing
├── models.ipynb                  # Model training, tuning and evaluation
├── streamlit_app.py                        # Streamlit web app
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

## Streamlit App

`app.py` is an interactive web app that estimates a patient's risk of heart disease using the Logistic Regression model.

**Try it live:** [Open the app on Streamlit Community Cloud](https://heart-disease-prediction-wwega8j3hbfpwrk8wd6xhq.streamlit.app/)

### Running the app

```bash
uv add streamlit   # if not already installed
uv run streamlit run streamlit_app.py
```

The app opens in your browser at `http://localhost:8501`. The `models/` folder must contain `logistic.joblib` and `preprocessor.joblib`, so run both notebooks first.

### How it works

1. The user enters the 13 clinical features (age, sex, chest pain type, blood pressure, cholesterol and so on).
2. The inputs are encoded and scaled with the saved preprocessor, using the same transformations as the training data.
3. The model returns the estimated probability of heart disease.
4. The result is shown as a **low**, **moderate** or **high** estimated risk band rather than a yes/no diagnosis, since a single cutoff makes nearby probabilities (e.g. 49% and 51%) look misleadingly different.

### Using a saved model in your own code

```python
import pandas as pd
from joblib import load

preprocess = load("models/preprocessor.joblib")
model = load("models/logistic.joblib")

raw = pd.DataFrame([{
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
    "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
    "ca": 0, "thal": 6.0,
}])

X = pd.DataFrame(preprocess.transform(raw), columns=preprocess.get_feature_names_out())
probability = model.predict_proba(X)[0, 1]
```

Inputs must use the dataset's coding: `cp` 1–4, `restecg` 0–2, `slope` 1–3 and `thal` 3, 6 or 7.

## Limitations

- The test set is small (84 patients), so differences of one or two patients between models are within noise.
- Continuous features were standardized before the train/test split, so a small amount of test-set information influenced preprocessing.
- Outlier removal may exclude clinically valid extreme values.

## Future Work

- Move preprocessing into a scikit-learn `Pipeline` fitted on training data only
- Compare models with repeated stratified cross-validation
- Tune the decision threshold to increase recall
- Add model interpretation (e.g. Logistic Regression coefficients or SHAP values)

## Disclaimer

This project is for educational purposes only and is not a medical diagnostic tool. The models were trained on a small historical dataset and their predictions should not be used to make health decisions. Anyone with concerns about their heart health should consult a qualified healthcare professional.

## Acknowledgements

Data: Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X
