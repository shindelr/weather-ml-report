"""
Author: Robin Shindelman
Date: 2025-05-01
CS 432 -- Applied Machine Learning

A use of logistic regression to predict whether it rained or not based on a few
variables.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

log = LogisticRegression()
fp = "/Users/robinshindelman/repos/weather-ml-report/data/clean/rain-occurence/balanced-clean-rain.csv"
df = pd.read_csv(fp)

X, y = train_test_split(df, test_size=.2)
X_labs, y_labs = X.Rain, y.Rain
X.drop('Rain', inplace=True, axis=1)
y.drop('Rain', inplace=True, axis=1)

log_mod = log.fit(X, X_labs)
print(f"Log Eqxn Numbers:\nIntercept = {log_mod.intercept_}\nCoeffs = {log_mod.coef_}")
log_preds = log_mod.predict(y)

print(df.iloc[3851])
print(f"Log Score: {log_mod.score(y, y_labs)}")

conf_m = confusion_matrix(y_true=y_labs, y_pred=log_preds)
plt.figure(figsize=(5,5))
sns.heatmap(conf_m, annot=True, cmap='Blues')
plt.title("Confusion Matrix")
plt.show()
