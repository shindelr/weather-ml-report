"""
Author: Robin Shindelman
Date: 2025-05-01
CS 432 -- Applied Machine Learning

A use of linear regression to predict urban air quality.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

fp = "/Users/robinshindelman/repos/weather-ml-report/data/clean/Urban Air Quality/urban_aqi_health_clean.csv"
df = pd.read_csv(fp)

to_remove = ['Unnamed: 0', 'datetime', 'sunrise', 'sunset', 'conditions', 'City',
             'Month', 'Day_of_Week', 'Is_Weekend']
df.drop(to_remove, inplace=True, axis=1)

lr = LinearRegression()
X, y = train_test_split(df, test_size=0.2)
X_labels, y_labels = X.Health_Risk_Score, y.Health_Risk_Score
X.drop("Health_Risk_Score", inplace=True, axis=1)
y.drop("Health_Risk_Score", inplace=True, axis=1)

lr.fit(X, X_labels)
print(y_labels.iloc[0])
print(f"LR Eqxn Numbers:\nIntercept = {lr.intercept_}\nCoeffs = {lr.coef_}")
print(f"LR Accuracy Score: {lr.score(y, y_labels)}")