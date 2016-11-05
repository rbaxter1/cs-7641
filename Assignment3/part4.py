import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from data_helper import *
from sklearn.datasets import make_blobs
from sklearn.preprocessing import Imputer, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.mixture import GaussianMixture, GMM
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import learning_curve, ShuffleSplit, StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from data_helper import *


from sklearn.mixture import GaussianMixture, GMM
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from plot_nn import plot_learning_curve, plot_validation_curve
from gridsearch_nn import run_grid_search

def nn_pca_cluster():
    dh = data_helper()
    X_train, X_test, y_train, y_test = dh.get_wine_data_all()
    
    X_train = MinMaxScaler().fit_transform(X_train)
    y_train = MinMaxScaler().fit_transform(y_train)
    
    # 3 from the orignal experiment
    pca = PCA(n_components=3, svd_solver='full')
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    
    # need to figure out the best clustering
    km = KMeans(n_clusters=3, algorithm='full')
    km.fit(X_train)
    X_train = km.predict(X_train)
    
    # hot encode
    enc = OneHotEncoder(sparse=False)
    enc.fit(X_train.reshape(-1, 1))
    
    enc.n_values_

    enc.feature_indices_
    X_train = enc.transform(X_train.reshape(-1, 1))
    
    
    title = "Learning Curves (Neural Network PCA KMeans)"
    
    clf = MLPClassifier(activation='relu',
                        learning_rate='constant',
                        shuffle=True,
                        solver='adam',
                        random_state=0,
                        max_iter=500,
                        batch_size=60)
    
    cv = StratifiedShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    
    out_dir = 'output_part4'
    name = 'nn_pca_km'
    fn = './' + out_dir + '/' + name + '.png'
    
    plot_learning_curve(clf, title, X_train, y_train, ylim=None, cv=cv, n_jobs=4, filename=fn)
    
    
def nn_pca():
    dh = data_helper()
    X_train, X_test, y_train, y_test = dh.get_wine_data_all()
    
    X_train = MinMaxScaler().fit_transform(X_train)
    y_train = MinMaxScaler().fit_transform(y_train)
    
    X_test = MinMaxScaler().fit_transform(X_test)
    y_test = MinMaxScaler().fit_transform(y_test)
    
    # 3 from the orignal experiment
    pca = PCA(n_components=3, svd_solver='full')
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    
    clf = MLPClassifier(activation='relu',
                        learning_rate='constant',
                        shuffle=True,
                        solver='sgd',
                        random_state=0,
                        max_iter=1000,
                        batch_size=60)
    
    cv = StratifiedShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    
    '''
    parameters = {'clf__alpha': (0.01, 0.001),
                  'clf__activation': ('identity', 'logistic', 'tanh', 'relu'),
                  'clf__learning_rate': ('constant', 'invscaling', 'adaptive'),
                  'clf__shuffle': (True, False),
                  'clf__learning_rate_init': (0.01, 0.001, 0.0001),
                  'clf__power_t': (0.0, 0.5, 1.0),
                  'clf__momentum': np.arange(0.0, 1.0, 0.1),
                  'clf__nesterovs_momentum': (True, False)
                  #'clf__max_iter': np.arange(1, 500, 10),
                  #'clf__batch_size': np.arange(50,500,10),
                  #'clf__learning_rate_init': np.arange(0.001,0.1,0.01),
                  #'clf__power_t': np.arange(0.01,0.1,0.01)
                  }
    '''
    
    parameters = {'learning_rate': ('constant', 'adaptive'),
                  'shuffle': (True, False),
                  'learning_rate_init': np.arange(0.001,0.1,0.01),
                  'momentum': np.arange(0.0, 1.0, 0.1),
                  'nesterovs_momentum': (True, False),
                  'early_stopping': (True, False)
                  }
    
    #run_grid_search(clf, parameters, X_train, X_test, y_train, y_test, cv)
    
    clf = MLPClassifier(activation='relu',
                        learning_rate='constant',
                        learning_rate_init=0.09,
                        momentum=0.9,
                        nesterovs_momentum=True,
                        shuffle=True,
                        solver='sgd',
                        random_state=0,
                        max_iter=1000,
                        batch_size=60,
                        early_stopping=False)
    
    
    '''
    Best score: 0.708
    Best parameters set:
        early_stopping: False
        learning_rate: 'adaptive'
        learning_rate_init: 0.090999999999999984
        momentum: 0.90000000000000002
        nesterovs_momentum: True
        shuffle: True

    '''
    
    '''
    Best score: 0.707
    Best parameters set:
            early_stopping: False
            learning_rate: 'constant'
            learning_rate_init: 0.090999999999999984
            momentum: 0.90000000000000002
            nesterovs_momentum: True
            shuffle: True
    Best parameters set found on development set:
    ()
    {'shuffle': True, 'nesterovs_momentum': True, 'learning_rate_init': 0.090999999999999984, 'learning_rate': 'constant', 'momentum': 0.90000000000000002, 'early_stopping': False}
    ()

    '''    
    
    '''
    clf = MLPClassifier(activation='relu',
                        learning_rate='invscaling',
                        shuffle=True,
                        solver='sgd',
                        random_state=0,
                        max_iter=1000,
                        batch_size=60)
    
    parameters = {'shuffle': (True, False),
                  'learning_rate_init': np.arange(0.001,0.1,0.01),
                  'momentum': np.arange(0.0, 1.0, 0.1),
                  'nesterovs_momentum': (True, False),
                  'early_stopping': (True, False),
                  'power_t': np.arange(0.01,0.1,0.01)
                  }
    
    run_grid_search(clf, parameters, X_train, X_test, y_train, y_test)
    '''
    
    
    
    
    
    
    title = "Validation Curves (Neural Network PCA)"
    out_dir = 'output_part4'
    name = 'nn_pca_vc_learning_rate'
    fn = './' + out_dir + '/' + name + '.png'
    
    '''
    plot_validation_curve(X_train,
                          X_test,
                          y_train,
                          y_test,
                          clf,
                          'momentum',
                          np.arange(0.0, 1.0, 0.1),
                          title,
                          fn,
                          param_range_plot=None)
    
    '''
    
    title = "Learning Curves (Neural Network PCA)"
    out_dir = 'output_part4'
    name = 'nn_ica'
    fn = './' + out_dir + '/' + name + '.png'
    
    plot_learning_curve(clf, title, X_train, y_train, ylim=None, cv=cv, n_jobs=4, filename=fn)
    
def nn_ica():
    dh = data_helper()
    X_train, X_test, y_train, y_test = dh.get_wine_data()
    
    scl = RobustScaler()
    X_train_scl = scl.fit_transform(X_train)
    X_test_scl = scl.transform(X_test)
    
    ica = FastICA(n_components=X_train_scl.shape[1])
    X_ica = ica.fit_transform(X_train_scl)
    
    ##
    ## ICA
    ##
    kurt = kurtosis(X_ica)
    print(kurt)
    i = kurt.argsort()[::-1]
     
    X_ica_sorted = X_ica[:, i]
    
    # top 3
    X_ica_top2 = X_ica_sorted[:,0:3]
    
    
    title = "Learning Curves (Neural Network ICA)"
    
    clf = MLPClassifier(activation='relu',
                        learning_rate='constant',
                        shuffle=True,
                        solver='adam',
                        random_state=0,
                        max_iter=500,
                        batch_size=60
                        )
    
    cv = StratifiedShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    
    out_dir = 'output_part4'
    name = 'nn_ica'
    fn = './' + out_dir + '/' + name + '.png'
    
    plot_learning_curve(clf, title, X_ica_top2, y_train, ylim=None, cv=cv, n_jobs=4, filename=fn)
    
    
    
def nn_orig():
    dh = data_helper()
    X_train, X_test, y_train, y_test = dh.get_wine_data()
    
    
    title = "Learning Curves (Neural Network Orig)"
    
    clf = Pipeline([('scl', MinMaxScaler()),
                    ('clf', MLPClassifier(activation='relu',
                                          learning_rate='constant',
                                          shuffle=True,
                                          solver='adam',
                                          random_state=0,
                                          max_iter=500,
                                          batch_size=60
                                          ))])
    
    cv = StratifiedShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    
    out_dir = 'output_part4'
    name = 'nn_orig'
    fn = './' + out_dir + '/' + name + '.png'
    
    plot_learning_curve(clf, title, X_train, y_train, ylim=None, cv=cv, n_jobs=4, filename=fn)
    
    

'''
4. Apply the dimensionality reduction algorithms to one of your datasets from assignment #1 (if you've reused
the datasets from assignment #1 to do experiments 1-3 above then you've already done this) and rerun your neural
network learner on the newly projected data.
'''
def main():
    
    print('Running part 4')
   
    # Apply the dimensionality reduction algorithms to one of your datasets from assignment #1
    #nn_orig()
    
    nn_pca()
    
    #nn_pca_cluster()
    
    # Rerun your neural network learner on the newly projected data.

if __name__== '__main__':
    main()
    
