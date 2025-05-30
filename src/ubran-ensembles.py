"""
author: Robin Shindelman
date: 2025-05-30

Ensemble classification.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import OrdinalEncoder, LabelBinarizer
from sklearn.tree import plot_tree

# Dataset loading
df = pd.read_csv('./data/clean/Urban Air Quality/urban_aqi_health_clean.csv')

# Binning label data
bin_names = ['Minimal', 'Low', 'Medium', 'High', 'Severe']
df['health_score_label'] = pd.cut(df.Health_Risk_Score, 5, labels=bin_names)
df = df.drop('Health_Risk_Score', axis=1)

non_quantitative = ['datetime', 'sunrise', 'sunset', 'conditions']
df = df.drop(non_quantitative, axis=1)
df = df.drop('Unnamed: 0', axis=1)

# Encoding Categorical Labels
to_be_ordinalized = ['City', 'Day_of_Week']
cities_codes = df.City.astype('category').cat.categories.to_list()
week_days_codes = df.Day_of_Week.astype('category').cat.categories.to_list()

ord_coder = OrdinalEncoder(categories=[cities_codes, week_days_codes])
df[to_be_ordinalized] = ord_coder.fit_transform(df[to_be_ordinalized])

# Encoding Binary Label
bin_coder = LabelBinarizer()
df['Is_Weekend'] = bin_coder.fit_transform(df['Is_Weekend'])

print(df)

# Dataset splitting
y = df['health_score_label']
X = df.drop('health_score_label', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

# ADABOOST SECTION
ada = AdaBoostClassifier(n_estimators=50, learning_rate=1.0)
ada_boost = ada.fit(X_train, y_train)
ada_preds = ada_boost.predict(X_test)
ada_classes = ada_boost.classes_
conf_m = confusion_matrix(y_true=y_test, y_pred=ada_preds, labels=ada_classes)
plt.figure(figsize=(5,5))
sns.heatmap(conf_m, annot=True, cmap='Blues', xticklabels=ada_classes, yticklabels=ada_classes, cbar=False)
plt.title("Adaboost Confusion Matrix")
plt.show()
print(f"Adaboost Score: {round(ada.score(X_test, y_test), 2)}\n{'-'*30}")

# RANDOM FOREST SECTION
rf = RandomForestClassifier(n_estimators=10, max_depth=5)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_classes = rf.classes_
conf_m = confusion_matrix(y_true=y_test, y_pred=rf_preds, labels=rf_classes)
plt.figure(figsize=(5,5))
sns.heatmap(conf_m, annot=True, cmap='Blues', xticklabels=rf_classes, yticklabels=rf_classes, cbar=False)
plt.title("Random Forest Confusion Matrix")
plt.show()
print(f"Random Forest Score: {round(rf.score(X_test, y_test), 2)}\n{'-'*30}")

# Visualizes three trees from the random forest
for i in range(3):
    next_tree = rf.estimators_[i]
    plt.figure(figsize=(12, 8))
    plot_tree(next_tree, feature_names=X_train.columns.values, 
          class_names=rf.classes_, filled=True)
    plt.show()

# STACKING SECTION
es = [
    ('rf', RandomForestClassifier(n_estimators=10)),
    ('svm', SVC(C=1., kernel='poly', degree=3)),
    ('linear', LinearRegression())
]
"""
I used these three models in my stacking ensemble. The first is a random forest,
the second a support vector machine, and the third is multinomial naive bayes. 
"""

stack = StackingClassifier(estimators=es, final_estimator=LogisticRegression())
stack.fit(X_train, y_train)
stack.score(X_test, y_test)
stack_preds = stack.predict(X_test)
stack_classes = stack.classes_
conf_m = confusion_matrix(y_true=y_test, y_pred=stack_preds, labels=stack_classes)
plt.figure(figsize=(5,5))
sns.heatmap(conf_m, annot=True, cmap='Blues', xticklabels=stack_classes, yticklabels=stack_classes, cbar=False)
plt.title("Stack Confusion Matrix")
plt.show()
print(f"Stacking Score: {round(stack.score(X_test, y_test), 2)}\n{'-'*30}")