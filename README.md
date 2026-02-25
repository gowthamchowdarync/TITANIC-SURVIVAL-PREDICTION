# Titanic Survival Prediction

## Project Overview
A complete machine learning pipeline to predict whether a Titanic passenger survived, using the classic Kaggle dataset.

## Files
| File | Description |
|------|-------------|
| `titanic_model.py` | Full ML pipeline script |
| `best_model.pkl` | Trained best model (Logistic Regression) |
| `scaler.pkl` | Fitted StandardScaler for preprocessing |
| `model_comparison.csv` | Accuracy comparison across all models |
| `titanic_analysis.png` | Full analysis & results visualisation |

## Results Summary

| Model | CV Accuracy | Test Accuracy | ROC AUC |
|-------|-------------|---------------|---------|
| Logistic Regression | 82.3% ± 2.9% | **83.8%** | **0.867** |
| SVM | 83.3% ± 2.5% | 82.7% | 0.849 |
| Random Forest | 82.3% ± 2.0% | 82.1% | 0.844 |
| Gradient Boosting | 81.3% ± 1.8% | 80.5% | 0.825 |

**Best Model:** Logistic Regression — Test Accuracy: **83.8%**, ROC AUC: **0.867**

## Feature Engineering
- **Title extraction** from passenger names (Mr, Mrs, Miss, Master, Other)
- **Age imputation** by median within each Title group
- **FamilySize** = SibSp + Parch + 1
- **IsAlone** flag for solo travellers
- **AgeBand** and **FareBand** (binned/quantile buckets)
- **HasCabin** flag
- One-hot encoding of Embarked and Title

## Top Predictive Features
1. Sex (female survival rate ~74% vs male ~19%)
2. Passenger class (1st class ~63% vs 3rd class ~24%)
3. Title (Mrs/Miss higher survival than Mr)
4. Fare paid
5. Age

## How to Use the Saved Model
```python
import joblib, pandas as pd

model  = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

# Prepare features in same order as training
features = ['Pclass', 'Sex_enc', 'Age', 'SibSp', 'Parch', 'Fare',
            'FamilySize', 'IsAlone', 'AgeBand', 'FareBand', 'HasCabin',
            'Emb_Q', 'Emb_S', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Other']

X_new = pd.DataFrame([{...}])   # fill in passenger data
X_scaled = scaler.transform(X_new[features])
prediction = model.predict(X_scaled)  # 0 = Died, 1 = Survived
```
