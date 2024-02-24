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
import matplotlib.animation as animation

from IPython.display import display
from rice_model_functions import convert_rice_class_to_number, normalize_dataframe

from sklearn.model_selection import train_test_split
from sklearn import datasets, decomposition
from sklearn import svm
from sklearn.metrics import accuracy_score

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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
X = df_normalized
y = pd.DataFrame(class_array_as_number, columns=["Class"])
y_reshaped = class_array_as_number.flatten().tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y_reshaped, test_size=0.33, random_state=42)

'''
clf = svm.SVC()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("svm accuracy: ",accuracy_score(y_test, y_pred))

clf = make_pipeline(StandardScaler(), SVC())
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("SVC accuracy: ",accuracy_score(y_test, y_pred))
'''

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

logistic = LogisticRegression(solver='saga', tol=1e-2, max_iter=200,random_state=0)
distributions = dict(C=uniform(loc=0, scale=4),penalty=['l2', 'l1'])
clf = RandomizedSearchCV(logistic, distributions, random_state=0)

search = clf.fit(X_train, y_train)
print(search.best_params_)

y_pred = clf.predict(X_test)

print("log accuracy: ",accuracy_score(y_test, y_pred))