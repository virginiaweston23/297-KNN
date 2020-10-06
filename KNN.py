import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Loading the prostate cancer dataset from the csv file using pandas
data = pd.read_csv('prostate-cancer-prediction.csv', header=0)
data.columns = ['ID', 'OUT', 'RAD', 'TEX', 'PERI', 'AREA', 'SMOO', 'COMP', 'SYMM', 'DIM']
features = ['RAD', 'TEX', 'PERI', 'AREA', 'SMOO', 'COMP', 'SYMM', 'DIM']
X = data[features]
y = data['OUT']
print('Class labels:', np.unique(y))
y_convert = pd.Series([0]*100)
for i in range(100):
    if y[i] == 'M':
        y_convert[i] = 1
y.update(y_convert)
y = y.astype('int').T
print('Class labels:', np.unique(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# Standardizing the features:
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# K-nearest neighbors
knn = KNeighborsClassifier(n_neighbors=5,
                           p=2,
                           metric='minkowski')
knn.fit(X_train_std, y_train)
y_pred = knn.predict(X_test_std)
print("KNN Accuracy: %3f" % accuracy_score(y_test, y_pred))
