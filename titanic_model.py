"""
Titanic Survival Prediction - Full ML Pipeline
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, roc_auc_score, roc_curve)
import joblib

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv('Titanic-Dataset.csv')
print("Dataset shape:", df.shape)
print("\nBasic info:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())
print("\nSurvival rate:", df['Survived'].mean().round(3))

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(data):
    df = data.copy()
    
    # Title extraction
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    title_map = {'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master'}
    df['Title'] = df['Title'].map(title_map).fillna('Other')
    
    # Age imputation by median per Title
    for title in df['Title'].unique():
        median_age = df.loc[df['Title'] == title, 'Age'].median()
        df.loc[(df['Age'].isnull()) & (df['Title'] == title), 'Age'] = median_age
    df['Age'].fillna(df['Age'].median(), inplace=True)
    
    # Fare imputation
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    
    # Embarked imputation
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    
    # Family size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Age bands
    df['AgeBand'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], 
                           labels=[0, 1, 2, 3, 4]).astype(int)
    
    # Fare bands
    df['FareBand'] = pd.qcut(df['Fare'], q=4, labels=[0, 1, 2, 3]).astype(int)
    
    # Has Cabin
    df['HasCabin'] = df['Cabin'].notna().astype(int)
    
    # Sex encoding
    df['Sex_enc'] = (df['Sex'] == 'female').astype(int)
    
    # Embarked encoding
    embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Emb', drop_first=True)
    df = pd.concat([df, embarked_dummies], axis=1)
    
    # Title encoding
    title_dummies = pd.get_dummies(df['Title'], prefix='Title', drop_first=True)
    df = pd.concat([df, title_dummies], axis=1)
    
    return df

df = engineer_features(df)

# Select features
feature_cols = ['Pclass', 'Sex_enc', 'Age', 'SibSp', 'Parch', 'Fare',
                'FamilySize', 'IsAlone', 'AgeBand', 'FareBand', 'HasCabin',
                'Emb_Q', 'Emb_S',
                'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Other']

# Only include columns that exist
feature_cols = [c for c in feature_cols if c in df.columns]
print("\nFeatures used:", feature_cols)

X = df[feature_cols]
y = df['Survived']

# ─────────────────────────────────────────────
# 3. SPLIT & SCALE
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 4. TRAIN MULTIPLE MODELS
# ─────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':        RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
    'Gradient Boosting':    GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42),
    'SVM':                  SVC(probability=True, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    X_tr = X_train_sc if name in ('Logistic Regression', 'SVM') else X_train
    cv_scores = cross_val_score(model, X_tr, y_train, cv=cv, scoring='accuracy')
    model.fit(X_tr, y_train)
    
    X_te = X_test_sc if name in ('Logistic Regression', 'SVM') else X_test
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    
    results[name] = {
        'model': model,
        'cv_mean': cv_scores.mean(),
        'cv_std':  cv_scores.std(),
        'test_acc': accuracy_score(y_test, y_pred),
        'roc_auc':  roc_auc_score(y_test, y_prob),
        'y_pred':   y_pred,
        'y_prob':   y_prob,
        'X_test':   X_te,
    }
    print(f"\n{name}:")
    print(f"  CV Accuracy : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"  ROC AUC     : {roc_auc_score(y_test, y_prob):.4f}")

# Best model
best_name = max(results, key=lambda k: results[k]['test_acc'])
best = results[best_name]
print(f"\n✅ Best model: {best_name} (Test Acc: {best['test_acc']:.4f})")

# ─────────────────────────────────────────────
# 5. VISUALISATIONS
# ─────────────────────────────────────────────
sns.set_style('whitegrid')
palette = {'survived': '#4CAF50', 'died': '#F44336'}

fig = plt.figure(figsize=(20, 24))
fig.suptitle('Titanic Survival Prediction — Analysis & Results', 
             fontsize=20, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

raw = pd.read_csv('Titanic-Dataset.csv')

# --- Plot 1: Overall survival ---
ax1 = fig.add_subplot(gs[0, 0])
counts = raw['Survived'].value_counts()
bars = ax1.bar(['Died', 'Survived'], counts.values, color=['#F44336', '#4CAF50'], edgecolor='white', width=0.5)
ax1.set_title('Overall Survival', fontsize=13, fontweight='bold')
ax1.set_ylabel('Count')
for bar, val in zip(bars, counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, str(val), ha='center', fontweight='bold')

# --- Plot 2: Survival by Sex ---
ax2 = fig.add_subplot(gs[0, 1])
sex_surv = raw.groupby('Sex')['Survived'].mean() * 100
ax2.bar(sex_surv.index, sex_surv.values, color=['#2196F3', '#E91E63'], edgecolor='white', width=0.5)
ax2.set_title('Survival Rate by Sex', fontsize=13, fontweight='bold')
ax2.set_ylabel('Survival Rate (%)')
for i, (idx, val) in enumerate(sex_surv.items()):
    ax2.text(i, val + 1, f'{val:.1f}%', ha='center', fontweight='bold')

# --- Plot 3: Survival by Pclass ---
ax3 = fig.add_subplot(gs[0, 2])
class_surv = raw.groupby('Pclass')['Survived'].mean() * 100
ax3.bar(['1st Class', '2nd Class', '3rd Class'], class_surv.values,
        color=['#FFD700', '#C0C0C0', '#CD7F32'], edgecolor='white', width=0.5)
ax3.set_title('Survival Rate by Ticket Class', fontsize=13, fontweight='bold')
ax3.set_ylabel('Survival Rate (%)')
for i, val in enumerate(class_surv.values):
    ax3.text(i, val + 1, f'{val:.1f}%', ha='center', fontweight='bold')

# --- Plot 4: Age distribution ---
ax4 = fig.add_subplot(gs[1, 0:2])
raw_age = raw.dropna(subset=['Age'])
ax4.hist(raw_age[raw_age['Survived']==0]['Age'], bins=30, alpha=0.6, color='#F44336', label='Died')
ax4.hist(raw_age[raw_age['Survived']==1]['Age'], bins=30, alpha=0.6, color='#4CAF50', label='Survived')
ax4.set_title('Age Distribution by Survival', fontsize=13, fontweight='bold')
ax4.set_xlabel('Age')
ax4.set_ylabel('Count')
ax4.legend()

# --- Plot 5: Fare distribution ---
ax5 = fig.add_subplot(gs[1, 2])
raw_fare = raw[raw['Fare'] < 300]
ax5.hist(raw_fare[raw_fare['Survived']==0]['Fare'], bins=30, alpha=0.6, color='#F44336', label='Died')
ax5.hist(raw_fare[raw_fare['Survived']==1]['Fare'], bins=30, alpha=0.6, color='#4CAF50', label='Survived')
ax5.set_title('Fare Distribution by Survival', fontsize=13, fontweight='bold')
ax5.set_xlabel('Fare (£)')
ax5.set_ylabel('Count')
ax5.legend()

# --- Plot 6: Model comparison ---
ax6 = fig.add_subplot(gs[2, 0])
names = list(results.keys())
accs  = [results[n]['test_acc'] for n in names]
colors = ['#42A5F5' if n != best_name else '#FF7043' for n in names]
bars = ax6.barh(names, accs, color=colors, edgecolor='white')
ax6.set_xlim(0.7, 1.0)
ax6.set_title('Model Test Accuracy', fontsize=13, fontweight='bold')
ax6.set_xlabel('Accuracy')
for bar, val in zip(bars, accs):
    ax6.text(val + 0.002, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center', fontsize=9)

# --- Plot 7: CV scores ---
ax7 = fig.add_subplot(gs[2, 1])
cv_means = [results[n]['cv_mean'] for n in names]
cv_stds  = [results[n]['cv_std']  for n in names]
ax7.barh(names, cv_means, xerr=cv_stds, color=colors, edgecolor='white', capsize=4)
ax7.set_xlim(0.7, 1.0)
ax7.set_title('5-Fold CV Accuracy (mean ± std)', fontsize=13, fontweight='bold')
ax7.set_xlabel('Accuracy')

# --- Plot 8: ROC curves ---
ax8 = fig.add_subplot(gs[2, 2])
roc_colors = ['#42A5F5', '#FF7043', '#66BB6A', '#AB47BC']
for (name, res), col in zip(results.items(), roc_colors):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    ax8.plot(fpr, tpr, color=col, lw=2, label=f"{name} (AUC={res['roc_auc']:.3f})")
ax8.plot([0,1],[0,1],'k--', lw=1)
ax8.set_title('ROC Curves', fontsize=13, fontweight='bold')
ax8.set_xlabel('False Positive Rate')
ax8.set_ylabel('True Positive Rate')
ax8.legend(fontsize=7)

# --- Plot 9: Confusion matrix (best model) ---
ax9 = fig.add_subplot(gs[3, 0])
cm = confusion_matrix(y_test, best['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax9,
            xticklabels=['Died', 'Survived'], yticklabels=['Died', 'Survived'])
ax9.set_title(f'Confusion Matrix\n({best_name})', fontsize=13, fontweight='bold')
ax9.set_ylabel('Actual')
ax9.set_xlabel('Predicted')

# --- Plot 10: Feature importance (best tree model) ---
ax10 = fig.add_subplot(gs[3, 1:])
rf = results['Random Forest']['model']
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)
importances.tail(12).plot(kind='barh', ax=ax10, color='#42A5F5', edgecolor='white')
ax10.set_title('Feature Importances (Random Forest)', fontsize=13, fontweight='bold')
ax10.set_xlabel('Importance')

plt.savefig('/home/claude/titanic_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved analysis chart.")

# ─────────────────────────────────────────────
# 6. SAVE BEST MODEL & SCALER
# ─────────────────────────────────────────────
best_model_obj = results[best_name]['model']
joblib.dump(best_model_obj, '/home/claude/best_model.pkl')
joblib.dump(scaler, '/home/claude/scaler.pkl')
print("Saved model and scaler.")

# ─────────────────────────────────────────────
# 7. PRINT FINAL REPORT
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("CLASSIFICATION REPORT — BEST MODEL:", best_name)
print("="*60)
print(classification_report(y_test, best['y_pred'], target_names=['Died', 'Survived']))

# ─────────────────────────────────────────────
# 8. SAVE SUMMARY CSV
# ─────────────────────────────────────────────
summary = pd.DataFrame([{
    'Model': n,
    'CV_Accuracy_Mean': f"{results[n]['cv_mean']:.4f}",
    'CV_Accuracy_Std':  f"{results[n]['cv_std']:.4f}",
    'Test_Accuracy':    f"{results[n]['test_acc']:.4f}",
    'ROC_AUC':          f"{results[n]['roc_auc']:.4f}",
} for n in names])
summary.to_csv('/home/claude/model_comparison.csv', index=False)
print("\nSaved model_comparison.csv")
print("\nDone!")
