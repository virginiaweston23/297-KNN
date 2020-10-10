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

# Standardize the features:
scaler = pp.PowerTransformer()
# scaler = pp.StandardScaler()
# scaler = pp.MaxAbsScaler()
# scaler = pp.RobustScaler()
scaler.fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

# K-nearest neighbors
knn_k = [5, 10, 15, 20, 25, 30, 35]
res_pred = []
res_times = []
for i in knn_k:
    t_b = time.perf_counter()
    knn = KNeighborsClassifier(n_neighbors=i, metric='manhattan', weights='distance')
    knn.fit(X_train_std, y_train)
    y_pred = knn.predict(X_test_std)
    res_pred.append(accuracy_score(y_test, y_pred))
    t_e = time.perf_counter()
    res_times.append((t_e - t_b)*1000)
plt.plot(knn_k, res_pred, 'r--', knn_k, res_times, 'b--')
plt.xlabel('K')
plt.ylabel('Accuracy (red) and Time (blue)')
plt.show()

# Best result given by n = 15, metric = 'manhattan', weights = 'distance'
knn_optimal = KNeighborsClassifier(n_neighbors=15, metric='manhattan', weights='distance')
knn_optimal.fit(X_train_std, y_train)
y_optimal_pred = knn_optimal.predict(X_test_std)
print("KNN Accuracy: {}".format(accuracy_score(y_test, y_optimal_pred)))
print("Metrics: {}".format(classification_report(y_test, y_optimal_pred)))