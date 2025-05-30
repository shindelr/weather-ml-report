"""
author: Robin Shindelman
date: 2025-03-05
description: Data processing for urban_aqi_health_clean.csv.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
from sklearn.preprocessing import OrdinalEncoder, LabelBinarizer


def load_csv(data_fp: str) -> pd.DataFrame:
    """ Load the .csv into a panda dataframe """
    return pd.read_csv(data_fp)

data_fp = 'data/clean/Urban Air Quality/urban_aqi_health_clean.csv'
df = load_csv(data_fp)
df = df.drop('Unnamed: 0', axis=1)

non_quantitative = ['datetime', 'sunrise', 'sunset', 'conditions']
df = df.drop(non_quantitative, axis=1)

# Encoding Categorical Labels
to_be_ordinalized = ['City', 'Day_of_Week']
cities_codes = df.City.astype('category').cat.categories.to_list()
week_days_codes = df.Day_of_Week.astype('category').cat.categories.to_list()

ord_coder = OrdinalEncoder(categories=[cities_codes, week_days_codes])
df[to_be_ordinalized] = ord_coder.fit_transform(df[to_be_ordinalized])

# Encoding Binary Label
bin_coder = LabelBinarizer()
df['Is_Weekend'] = bin_coder.fit_transform(df['Is_Weekend'])

# Binning label data
bin_names = ['Minimal', 'Low', 'Medium', 'High', 'Severe']
df['health_score_label'] = pd.cut(df.Health_Risk_Score, 5, labels=bin_names)
df = df.drop('Health_Risk_Score', axis=1)

# Train-test split
X = df.drop('health_score_label', axis=1)
y = df.health_score_label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,  random_state=12)

# Decision Tree
# for i in range(2, 50):
clf = DecisionTreeClassifier(random_state=3, 
                                max_depth=8, 
                                max_features=7,
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
plt.show()

plt.figure(figsize=(12,12))
plot_tree(tree_mod, 
          feature_names=tree_mod.feature_names_in_,
          class_names=tree_mod.classes_,
          filled=True)
plt.savefig("plots/aqi-tree-plot2.svg")
plt.close()

