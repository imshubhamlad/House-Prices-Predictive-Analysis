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

The project evolved through multiple modelling experiments. The final V3 workflow focuses on improving data quality and making model selection data-driven.

### Key improvements in V3

- Removed `Id` from the modelling features because it does not represent a meaningful house characteristic.
- Treated `MSSubClass` as categorical rather than numeric.
- Added scaling for regularized linear models such as Ridge and Lasso.
- Imputed `LotFrontage` using neighborhood-level information rather than a global median.
- Applied `log1p` transformations to highly skewed numerical predictors.
- Checked and removed known extreme `GrLivArea` observations that could distort model fitting.
- Compared models using cross-validation before tuning the selected model.
- Added XGBoost and a simple model blend as part of the final experimentation.

## Model Evaluation

The Kaggle leaderboard metric is based on the RMSE of the logarithm of predicted and actual sale prices.

The final V3 notebook reports a Kaggle score of **0.12506**, improving on the earlier V1 score of **0.14337** and V2 score of **0.14663**.

## Project Structure

```text
House-Prices-Predictive-Analysis/
├── House_Price_Model_V3.ipynb          # Main modelling notebook
├── House_price_regression_v1_0.ipynb   # Earlier experiment
├── Regression Model V2_Failed.ipynb    # Earlier experiment / learning iteration
├── train.csv                           # Training dataset
├── test.csv                            # Testing dataset
├── sample_submission.csv               # Kaggle submission format
├── submission_v1.csv                   # Earlier submission
├── submissionV2.csv                    # Earlier submission
└── submissionV3.csv                    # Latest submission
```

## Key Learning

The project demonstrates that model performance depends not only on the choice of algorithm, but also on data preparation, feature representation, validation strategy, and disciplined model selection. The progression from V1 to V3 was used to identify sources of noise and improve the overall modelling pipeline.

## Technologies

Python · Pandas · NumPy · Scikit-learn · Seaborn · Matplotlib · SciPy · XGBoost · Jupyter Notebook

## Author

**Shubham Lad**
