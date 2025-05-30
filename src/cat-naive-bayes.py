"""
author: Robin Shindelman
date: 2025-05-16
description: Data processing for naive bayes on AQI
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.naive_bayes import CategoricalNB

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import OrdinalEncoder, LabelBinarizer

def load_csv(data_fp: str) -> pd.DataFrame:
    """ Load the .csv into a panda dataframe """
    return pd.read_csv(data_fp)

data_fp = 'data/clean/Urban Air Quality/urban_aqi_health_clean.csv'
df = load_csv(data_fp)
df = df.drop('Unnamed: 0', axis=1)

# Dropping quantitative columns
df.drop(columns=df.loc[:, "datetime":'sunset'].columns, inplace=True)
df.drop(columns=df.loc[:, "Temp_Range":'Severity_Score'].columns, inplace=True)

# Encoding Categorical Labels
to_be_ordinalized = ['conditions', 'City', 'Day_of_Week']
cities_codes = df.City.astype('category').cat.categories.to_list()
week_days_codes = df.Day_of_Week.astype('category').cat.categories.to_list()
conditions_codes = df.conditions.astype('category').cat.categories.to_list()

ord_coder = OrdinalEncoder(categories=[conditions_codes, cities_codes, week_days_codes])
df[to_be_ordinalized] = ord_coder.fit_transform(df[to_be_ordinalized])

# Encoding Binary Label
bin_coder = LabelBinarizer()
df['Is_Weekend'] = bin_coder.fit_transform(df['Is_Weekend'])

# Binning label data
bin_names = ['Minimal', 'Low', 'Medium', 'High', 'Severe']
df['health_score_label'] = pd.cut(df.Health_Risk_Score, 5, labels=bin_names)
df = df.drop('Health_Risk_Score', axis=1)

# Train-test split
X = df.drop("health_score_label", axis=1)
y = df["health_score_label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

# Model
cat = CategoricalNB()
cat_mod = cat.fit(X_train, y_train)
cat_preds = cat_mod.predict(X_test)
print(f"Categoricalgit  Probs:\n{cat_mod.predict_proba(X_test).round(3)}\n")
print(f"{'-'*10}Validation{'-'*10}\n{cat_mod.score(X_test, y_test)}")
cat_cm = confusion_matrix(y_true=y_test, y_pred=cat_preds)
plt.figure(figsize=(5,5))
sns.heatmap(cat_cm, 
            annot=True, 
            cmap='Blues', 
            xticklabels=bin_names, 
            yticklabels=bin_names, 
            cbar=False)
plt.title('Confusion Matrix for Urban Health Risk Score Predictions')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()