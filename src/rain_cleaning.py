"""
author: Robin Shindelman
date: 2025-02-27
description: Data processing for rain_or_norain.csv.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample


def load_csv(data_fp: str) -> pd.DataFrame:
    """ Load the .csv into a panda dataframe """
    return pd.read_csv(data_fp)

data_fp = '../data/raw/rain/rain_or_norain.csv'
df = load_csv(data_fp)
print(df.head())
print(df.info())

# Visualizations 
sns.countplot(data=df, x="Rain")
plt.title("Distribution of Labels")
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, col in enumerate(df.columns):
    if i <= 1:
        sns.histplot(data=df, x=col, ax=axes[i])
    if i > 1 and i < 4:
        sns.boxplot(data=df, x=col, ax=axes[i])
fig.suptitle('Distribution of weather variables')
plt.tight_layout()
plt.show()

# sns.catplot(data=df, x='Cloud_Cover', y='Rain', hue='Pressure', kind='swarm', size=3)
# plt.title('Swarmplot of Cloud Cover and Pressure effect on the label')
# plt.tight_layout()
# plt.show()

print(f"Dataset imbalance: {(((df.Rain == 'rain').sum()) / len(df)) * 100}%")
print("-- Upsampling now --")
minority = df[df.Rain == 'rain']
majority = df[df.Rain == 'no rain']
minor_upsample = resample(minority, replace=True, n_samples=len(majority), random_state=12)
df_balanced = pd.concat([majority, minor_upsample])

sns.countplot(data=df_balanced, x="Rain")
plt.title("Distribution of Labels, Balanced")
plt.show()

df_balanced.to_csv('../data/clean/rain-occurence/balanced-clean-rain.csv', index=False)