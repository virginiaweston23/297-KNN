import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix
import numpy as np
from mlxtend.plotting import heatmap

# Load dataset
data = pd.read_csv('prostate-cancer-prediction.csv', header=0)
data.columns = ['ID', 'OUT', 'RAD', 'TEX', 'PERI', 'AREA', 'SMOO', 'COMP', 'SYMM', 'DIM']
features = ['RAD', 'TEX', 'PERI', 'AREA', 'SMOO', 'COMP', 'SYMM', 'DIM']
X = data[features]
y = data['OUT']

# EDA
cm = np.corrcoef(data[features].values.T)
hm = heatmap(cm, row_names=features, column_names=features)

scatterplotmatrix(data[features].values, figsize=(10, 8), names=data.columns, alpha=0.5)
plt.show()
