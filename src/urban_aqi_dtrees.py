"""
author: Robin Shindelman
date: 2025-03-05
description: Data processing for urban_aqi_health_clean.csv.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics


def load_csv(data_fp: str) -> pd.DataFrame:
    """ Load the .csv into a panda dataframe """
    return pd.read_csv(data_fp)

data_fp = 'data/clean/Urban Air Quality/urban_aqi_health_clean.csv'
df = load_csv(data_fp)
df = df.drop('Unnamed: 0', axis=1)

non_quantitative = ['datetime', 'sunrise', 'sunset', 'conditions', 
                    'City', 'Day_of_Week', 'Is_Weekend']
df = df.drop(non_quantitative, axis=1)

# Binning label data
bin_names = ['Minimal', 'Low', 'Medium', 'High', 'Severe']
df['health_score_label'] = pd.cut(df.Health_Risk_Score, 5, labels=bin_names)
df = df.drop('Health_Risk_Score', axis=1)

# Visualize Training data
plt.figure(figsize=(5,5))
sns.countplot(df, x='health_score_label')
plt.title("Distribution of training labels")
plt.xlabel("Label")

plt.figure(figsize=(5,5))
sns.scatterplot(df, x='temp', y='Heat_Index', hue='humidity')
plt.title("Heat Index by Temperature and Humidity")
plt.xlabel('Temperature')
plt.ylabel('Heat Index')
plt.legend(loc='upper left', title='Humidity')

plt.figure(figsize=(8,5))
sns.swarmplot(data=df, 
              x='health_score_label', 
              y='Severity_Score',
              hue='feelslike',
              size=4)
plt.title('Beeswarm plot of Label vs. Weather Severity and "Feels Like" Temperature')
plt.xlabel('Label')
plt.ylabel('Weather Severity Index')
plt.legend(title='Feels-Like')
# plt.show()

# Train-test split
X = df.drop('health_score_label', axis=1)
y = df.health_score_label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,  random_state=12)

# Decision Tree
# for i in range(2, 50):
clf = DecisionTreeClassifier(random_state=3, 
                                max_depth=15, 
                                max_features=14,
                                min_samples_split=8,)
                                # max_leaf_nodes=10)
tree_mod = clf.fit(X_train, y_train)
preds = tree_mod.predict(X_test)

acc = metrics.accuracy_score(y_test, preds)
print("\n------------- Validation ------------- ")
print(f"Training Accuracy: {acc}")
print(f'Max Features: {clf.max_features_}')
print(f'Depth: {tree_mod.get_depth()}')

plt.figure(figsize=(8,8))
confusion_matrix = metrics.confusion_matrix(y_test, preds)
sns.heatmap(confusion_matrix, 
            annot=True, 
            cmap='Blues', 
            xticklabels=bin_names, 
            yticklabels=bin_names, 
            cbar=False)
plt.title('Confusion Matrix for Urban Health Risk Score Predictions')
plt.xlabel('Actual')
plt.ylabel('Predicted')
# plt.show()

plt.figure(figsize=(12,12))
plot_tree(tree_mod, 
          feature_names=tree_mod.feature_names_in_,
          class_names=tree_mod.classes_,
          filled=True)
plt.savefig("plots/aqi-tree-plot.svg")
plt.close()

