"""
02/07/2024
train the Rice Model using sklearn
"""

import sklearn
import pandas as pd
import numpy as np
from scipy.io.arff import loadarff
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

from IPython.display import display
from rice_model_functions import convert_rice_class_to_number, normalize_dataframe

from sklearn.model_selection import train_test_split
from sklearn import datasets, decomposition

# Get data
raw_data = loadarff('Rice_Cammeo_Osmancik.arff')
df_raw_data = pd.DataFrame(raw_data[0])

# Scramble the data rows
df_scrambled = df_raw_data.sample(frac=1, random_state=2)

# Change all data to numbers, specifically the class column
class_array_as_number = convert_rice_class_to_number(df_scrambled)

# Normalize all the data
df_normalized = normalize_dataframe(df_scrambled.drop("Class", axis=1))

## Specify Training and Testing Data
# Get trainning data as numpy array
#X = df_normalized.to_numpy()
#y = class_array_as_number
X_raw = df_normalized
y = pd.DataFrame(class_array_as_number, columns=["Class"])

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

y_reshaped = class_array_as_number.flatten().tolist()

## PCA
fig = plt.figure(1, figsize=(4, 3))
plt.clf()

ax = fig.add_subplot(111, projection="3d", elev=48, azim=134)
ax.set_position([0, 0, 0.95, 1])


plt.cla()
pca = decomposition.PCA(n_components=3)
pca.fit(X_raw)
X = pca.transform(X_raw)

cammeo_center = X[np.where(np.array(y_reshaped) == 0)].mean(axis=0)
osmancik_center = X[np.where(np.array(y_reshaped) == 1)].mean(axis=0)

ax.text(cammeo_center[0], cammeo_center[1], cammeo_center[2], "Cammeo", bbox=dict(alpha=0.5, edgecolor="w", facecolor="w"))
ax.text(osmancik_center[0], osmancik_center[1], cammeo_center[2], "Osmancik", bbox=dict(alpha=0.5, edgecolor="w", facecolor="w"))

# Reorder the labels to have colors matching the cluster results
colors = np.choose(y_reshaped, ["orange", "blue"])
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors, cmap=plt.cm.nipy_spectral, edgecolor="k")

ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])

#plt.show()

#print(pd.DataFrame(pca.components_,columns=X_raw.columns,index = ['PC-1','PC-2','PC-3']))