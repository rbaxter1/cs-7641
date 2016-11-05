print(__doc__)

# Authors: Kyle Kastner
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from data_helper import *
iris = load_iris()
X = iris.data
y = iris.target

dh = data_helper()
X_train, X_test, y_train, y_test = dh.get_wine_data()

# alternately MinMaxScaler
scaler = RobustScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X = X_train
y = y_train
y.shape = (y.shape[0],)

n_components = 2
ipca = IncrementalPCA(n_components=n_components, batch_size=10)
X_ipca = ipca.fit_transform(X)

pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

colors = ['red', 'green']
target_names = ['Worst', 'Best']
markers = ['<', 'v']
for X_transformed, title in [(X_ipca, "Incremental PCA"), (X_pca, "PCA")]:
    plt.figure(figsize=(8, 8))
    for color, i, target_name, marker in zip(colors, [0, 1], target_names, markers):
        plt.scatter(X_transformed[y == i, 0.], X_transformed[y == i, 1.],
                    color=color, lw=2, label=target_name, alpha=.5, marker=marker, facecolors='none')

    if "Incremental" in title:
        err = np.abs(np.abs(X_pca) - np.abs(X_ipca)).mean()
        plt.title(title + " of iris dataset\nMean absolute unsigned error "
                  "%.6f" % err)
    else:
        plt.title(title + " of iris dataset")
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.axis([-4, 4, -1.5, 1.5])

plt.show()