import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm

from sklearn.preprocessing import MinMaxScaler, StandardScaler, Imputer, OneHotEncoder, RobustScaler
from sklearn.decomposition import PCA, FastICA, RandomizedPCA, IncrementalPCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import cross_validation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_validation import KFold
from sklearn.metrics import *

from scipy.stats import kurtosis

from timeit import default_timer as time

from data_helper import *
from plot_helper import *

from part4 import *

'''
5. Apply the clustering algorithms to the same dataset to which you just applied the dimensionality
reduction algorithms (you've probably already done this), treating the clusters as if they were new
features. In other words, treat the clustering algorithms as if they were dimensionality reduction
algorithms. Again, rerun your neural network learner on the newly projected data.
'''        
class part5():
    def __init__(self):
        self.out_dir = 'output_part5'
        self.part4 = part4()
        self.part4.out_dir = self.out_dir

    def run(self):
        print('Running part 5')
    
        filename = './' + self.out_dir + '/time.txt'
        with open(filename, 'w') as text_file:
            
            t0 = time()
            self.nn_pca_cluster_wine()
            text_file.write('nn_pca_wine: %0.3f seconds\n' % (time() - t0))
            
            t0 = time()
            self.nn_ica_cluster_wine()
            text_file.write('nn_ica_wine: %0.3f seconds\n' % (time() - t0))
            
            t0 = time()
            self.nn_rp_cluster_wine()
            text_file.write('nn_rp_wine: %0.3f seconds\n' % (time() - t0))
            
            t0 = time()
            self.nn_lda_cluster_wine()
            text_file.write('nn_lda_wine: %0.3f seconds\n' % (time() - t0))
            
        
    def nn_pca_cluster_wine(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_wine_data_pca_best()
        self.part4.nn_analysis(X_train, X_test, y_train, y_test, 'Wine', 'Neural Network PDA')
        
    def nn_ica_cluster_wine(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_wine_data_ica_best()
        self.part4.nn_analysis(X_train, X_test, y_train, y_test, 'Wine', 'Neural Network IDA')
        
    def nn_rp_cluster_wine(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_wine_data_rp_best()
        self.part4.nn_analysis(X_train, X_test, y_train, y_test, 'Wine', 'Neural Network RP')
        
    def nn_lda_wine(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_wine_data_lda_best()
        self.part4.nn_analysis(X_train, X_test, y_train, y_test, 'Wine', 'Neural Network LDA')
    
def main():    
    p = part4()
    p.run()

if __name__== '__main__':
    main()
    
