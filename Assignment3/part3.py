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
from plot_nn import plot_learning_curve
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


'''
3. Reproduce your clustering experiments, but on the data after you've run dimensionality reduction on it.
'''
def main():
    
    print('Running part 3')
   
    # Apply the dimensionality reduction algorithms to one of your datasets from assignment #1
    #nn_orig()
    
    #nn_pca()
    
    nn_pca_cluster()
    
    # Rerun your neural network learner on the newly projected data.

if __name__== '__main__':
    main()
    
