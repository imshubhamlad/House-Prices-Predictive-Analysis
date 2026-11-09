# House Price Prediction — Machine Learning Regression

An end-to-end machine learning project for predicting residential house prices using the Kaggle **House Prices: Advanced Regression Techniques** dataset.

## Objective

Build and compare regression models, improve the preprocessing pipeline through feature engineering and cross-validation, and generate predictions for the Kaggle test set.

## Dataset

- Training data: 1,460 rows and 81 columns
- Test data: 1,459 rows and 80 columns
- Target: `SalePrice`

## Exploratory Data Analysis (EDA)

The initial analysis focused on understanding the target variable, identifying important predictors, assessing missing values, checking skewness, and detecting potential outliers.

### Key observations

- **SalePrice is right-skewed**, with a relatively small number of expensive houses pulling the distribution toward higher values. This supported the use of a logarithmic transformation for the target during modelling.
- **OverallQual** showed a strong relationship with `SalePrice`, indicating that overall material and finish quality is an important predictor of house value.
- **GrLivArea** showed a strong positive relationship with `SalePrice`, while also revealing a small number of unusually large houses that could disproportionately influence regression models.
- Several numerical variables were **highly skewed**, motivating the use of `log1p` transformations during preprocessing.
- The dataset contained **substantial missing values** in some features. Missingness was investigated before deciding how each variable should be handled rather than applying one blanket imputation strategy.
- Categorical variables such as `Neighborhood`, `ExterQual`, `KitchenQual`, and other quality-related features showed meaningful differences in house prices across their categories.
- Correlation analysis helped identify variables with strong relationships with the target and guided further feature investigation.

### How EDA influenced the modelling

The EDA was not treated as a separate visualization exercise. The findings directly influenced the V3 pipeline:

| EDA finding | Modelling decision |
|---|---|
| Right-skewed numerical variables | Applied `log1p` transformations |
| Missing `LotFrontage` values | Used neighborhood-level median imputation |
| `MSSubClass` represents a coded category | Treated it as categorical |
| Extreme `GrLivArea` observations | Investigated and removed known outliers |
| Strong categorical effects | Used categorical encoding |
| Different feature scales | Added scaling for Ridge/Lasso |
| Potential model sensitivity | Used cross-validation for model comparison |

The EDA therefore helped move the project from simply fitting models to building a more appropriate and robust preprocessing pipeline.

## Approach

The project evolved through multiple modelling experiments. V3 focused on fixing the preprocessing pipeline and making model selection data-driven. **V4 builds directly on V3's own "Where to go from here" notes** and focuses on getting more out of the model layer: benchmarking a second gradient-boosted model, and replacing a hand-weighted blend with a proper stacking ensemble.

### Key improvements in V3

- Removed `Id` from the modelling features because it does not represent a meaningful house characteristic.
- Treated `MSSubClass` as categorical rather than numeric.
- Added scaling for regularized linear models such as Ridge and Lasso.
- Imputed `LotFrontage` using neighborhood-level information rather than a global median.
- Applied `log1p` transformations to highly skewed numerical predictors.
- Checked and removed known extreme `GrLivArea` observations that could distort model fitting.
- Compared models using cross-validation before tuning the selected model.
- Added XGBoost and a simple model blend as part of the final experimentation.

### Key improvements in V4

- Added two interaction features: `OverallGrade` (`OverallQual * OverallCond`) and `QualAreaScore` (`OverallQual * GrLivArea`), to capture quality/condition and quality/size effects that neither raw column captures alone.
- Added **ElasticNet** and **LightGBM** to the model comparison, so the full space of reasonable candidates is benchmarked before anything is tuned.
- Tuned the best *linear* model and the best *tree-based* model separately (rather than only the single overall CV winner), because the stacking step needs two models that make different kinds of errors.
- Replaced V3's fixed inverse-RMSE blend with a `StackingRegressor`: a Ridge meta-learner trained on out-of-fold base-model predictions, so the blend weights are learned from data instead of hand-computed from a single CV score.
- Verified, rather than assumed, that stacking genuinely diverse models (a regularized linear model + a tree ensemble) beats stacking two correlated linear models — an early attempt at stacking Lasso/Ridge/ElasticNet together actually underperformed the single best linear model.

## Model Evaluation

The Kaggle leaderboard metric is based on the RMSE of the logarithm of predicted and actual sale prices. The notebooks' own 5-fold cross-validation RMSE is measured on the same log scale, so CV and leaderboard numbers are directly comparable.

| Version | CV / Kaggle RMSE | Approach |
|---|---|---|
| V1 | 0.14337 | Baseline regression |
| V2 | 0.14663 | Regression attempt with preprocessing bugs (regressed vs. V1) |
| V3 | 0.12506 (Kaggle) | Fixed preprocessing + XGBoost/Lasso blend |
| **V4** | **~0.1075 (CV)** | Fixed preprocessing + ElasticNet/Lasso + XGBoost **stacked** with a Ridge meta-learner |

V4 was validated with 5-fold cross-validation during development; it has not yet been scored on the Kaggle leaderboard.

## Project Structure

```text
House-Prices-Predictive-Analysis/
├── House_Price_Model_V4.ipynb          # Latest modelling notebook (stacked ensemble)
├── House_Price_Model_V3.ipynb          # Previous modelling notebook
├── House_price_regression_v1_0.ipynb   # Earlier experiment
├── Regression Model V2_Failed.ipynb    # Earlier experiment / learning iteration
├── train.csv                           # Training dataset
├── test.csv                            # Testing dataset
├── sample_submission.csv               # Kaggle submission format
├── submission_v1.csv                   # Earlier submission
├── submissionV2.csv                    # Earlier submission
├── submissionV3.csv                    # V3 submission
└── submissionV4.csv                    # Latest submission
```

## Key Learning

The project demonstrates that model performance depends not only on the choice of algorithm, but also on data preparation, feature representation, validation strategy, and disciplined model selection. The progression from V1 to V3 was used to identify sources of noise and improve the overall modelling pipeline. V4 added a further lesson: an ensemble only improves on its best member when its components are genuinely diverse — stacking two correlated linear models produced no gain, while stacking a regularized linear model with a tree-based model did.

## Technologies

Python · Pandas · NumPy · Scikit-learn · Seaborn · Matplotlib · SciPy · XGBoost · LightGBM · Jupyter Notebook

## Author

**Shubham Lad**
