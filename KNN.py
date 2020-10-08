import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as pp
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import time


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
sc = pp.PowerTransformer()
# sc = pp.StandardScaler()
# sc = pp.MaxAbsScaler()
# sc = pp.RobustScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# K-nearest neighbors
# Best result given by n = 25, metric = 'manhattan'
knn_n = [5, 10, 15, 20, 25, 30, 35]
for i in knn_n:
    t_b = time.perf_counter()
    knn = KNeighborsClassifier(n_neighbors=i, metric='manhattan', weights='distance')
    knn.fit(X_train_std, y_train)
    y_pred = knn.predict(X_test_std)
    t_e = time.perf_counter()
    print("KNN Accuracy: {} when n = {}".format(accuracy_score(y_test, y_pred), i))
    print((t_e - t_b)*1000)

knn_optimal = KNeighborsClassifier(n_neighbors=30, metric='manhattan', weights='distance')
knn_optimal.fit(X_train_std, y_train)
y_optimal_pred = knn_optimal.predict(X_test_std)
print("KNN Accuracy: {}".format(accuracy_score(y_test, y_optimal_pred)))
print("Other Metrics: {}".format(classification_report(y_test, y_optimal_pred)))