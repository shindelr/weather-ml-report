"""
author: Robin Shindelman
date: 2025-04-18
description: Clustering for the rain occurence dataset.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

df = pd.read_csv('/Users/robinshindelman/repos/weather-ml-report/data/clean/rain-occurence/balanced-clean-rain.csv')
og_labels = df.Rain
df.drop('Rain', inplace=True, axis=1)

# ks = [1, 2, 3]
# for k in ks:
#     kmeans = KMeans(n_clusters=k)
#     fitted = kmeans.fit(df)
#     clust_lables = fitted.labels_
#     if k <= 2:
#         plt.figure()
#         plt.title(f'K-Means Performed With k={k}')
#         sns.scatterplot(data=df, x='Cloud_Cover', y='Humidity', hue=clust_lables)
#         plt.xlabel('Cloud Cover')
#         plt.ylabel('Humidity')
#     else:
#         df['Class'] = og_labels
#         colors = {'rain': 'blue', 'no rain': 'orange'}
#         df["Color"] = df["Class"].map(colors)
#         fig3d = plt.figure()
#         ax2 = fig3d.add_subplot(projection='3d')
#         ax2.scatter(df.Cloud_Cover, df.Humidity, df.Wind_Speed, cmap='RdYlGn', edgecolor='k', s=200, c=df.Color)
#         ax2.set_xlabel('Cloud Cover')
#         ax2.set_ylabel('Humidity')
#         ax2.set_zlabel('Wind Speed')
#         ax2.set_title(f'K-Means Performed With k={k}')
#         for cls, color in colors.items():
#             ax2.scatter([], [], [], c=color, label=cls)
#         ax2.legend(title="Class")
#     plt.show()

pca = PCA(n_components=3)
kmeans = KMeans(n_clusters=2)
np_3d = pca.fit_transform(df)
df_3d = pd.DataFrame(np_3d, columns=["PC1", "PC2", 'PC3'])
print(df_3d)
fitted = kmeans.fit(df_3d)

# df['Class'] = og_labels
# colors = {'rain': 'blue', 'no rain': 'orange'}
# df["Color"] = df["Class"].map(colors)
# fig3d = plt.figure()
# ax2 = fig3d.add_subplot(projection='3d')
# ax2.scatter(df_3d.PC1, df_3d.PC2, df_3d.PC3, cmap='RdYlGn', edgecolor='k', s=200, c=df.Color)
# ax2.set_xlabel('PC1')
# ax2.set_ylabel('PC2')
# ax2.set_zlabel('PC3')
# ax2.set_title(f'K-Means Performed With k=2 and 3D PCA Reduction')
# for cls, color in colors.items():
#     ax2.scatter([], [], [], c=color, label=cls)
# ax2.legend(title="Class")
# plt.show()

hac = AgglomerativeClustering()
clusters = hac.fit(df_3d)
Z = linkage(df_3d)
plt.figure(figsize=(25, 10))
plt.title('Dendrogram of Rain Occurence')
plt.xlabel("Leaf Counts")
plt.ylabel("Cluster Distance")
dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=48,  # show only the last p merged clusters
    leaf_rotation=90.,  
    leaf_font_size=8.,  
)
plt.show()