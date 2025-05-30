"""
author: Robin Shindelman
date: 2025-05-30

SVM classification.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


fp = "data/clean/rain-occurence/balanced-clean-rain.csv"
df = pd.read_csv(fp)

y = df.Rain
X = df.drop('Rain', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

svm = SVC(C=1, kernel='linear')
svm.fit(X_train, y_train)
classes = svm.classes_
preds = svm.predict(X_test)
conf_m = confusion_matrix(y_true=y_test, y_pred=preds, labels=classes)
plt.figure(figsize=(5,5))
sns.heatmap(conf_m, 
            annot=True, 
            fmt = 'd',
            cmap='Blues', 
            xticklabels=classes, 
            yticklabels=classes, 
            cbar=False)
plt.title(f"Confusion Matrix -- Linear -- Cost=1")
plt.show()

print(svm.score(X_test, y_test))