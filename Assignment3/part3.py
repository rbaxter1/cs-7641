import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import itertools

from sklearn.preprocessing import MinMaxScaler, StandardScaler, Imputer, OneHotEncoder, RobustScaler
from sklearn.decomposition import PCA, FastICA, RandomizedPCA, IncrementalPCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import *

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from timeit import default_timer as time

from data_helper import *
from plot_helper import *
from part1 import *

'''
3. Reproduce your clustering experiments, but on the data after you've run dimensionality reduction on it.
'''
class part3():
    def __init__(self):
        self.save_dir = 'output_part3'
        self.part1 = part1()
        self.part1.out_dir = self.save_dir
    
    def kmeans_pca_wine(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_wine_data_pca_best()
        self.part1.kmeans_analysis(X_train, X_test, y_train, y_test, 'Wine', 20, 'GMM PCA')
    
    def kmeans_pca_nba(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_nba_data_pca_best()
        self.part1.kmeans_analysis(X_train, X_test, y_train, y_test, 'NBA', 20, 'GMM PCA')
        
    def kmeans_ica_wine(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_wine_data_ica_best()
        self.part1.kmeans_analysis(X_train, X_test, y_train, y_test, 'Wine', 20, 'GMM ICA')
    
    def kmeans_ica_nba(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_nba_data_ica_best()
        self.part1.kmeans_analysis(X_train, X_test, y_train, y_test, 'NBA', 20, 'GMM ICA')
        
    def kmeans_rp_wine(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_wine_data_rp_best()
        self.part1.kmeans_analysis(X_train, X_test, y_train, y_test, 'Wine', 20, 'GMM RP')
    
    def kmeans_rp_nba(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_nba_data_rp_best()
        self.part1.kmeans_analysis(X_train, X_test, y_train, y_test, 'NBA', 20, 'GMM RP')
        
    def kmeans_lda_wine(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_wine_data_lda_best()
        self.part1.kmeans_analysis(X_train, X_test, y_train, y_test, 'Wine', 20, 'GMM LDA')
    
    def kmeans_lda_nba(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_nba_data_lda_best()
        self.part1.kmeans_analysis(X_train, X_test, y_train, y_test, 'NBA', 20, 'GMM LDA')
        
    def gmm_pca_wine(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_wine_data_pca_best()
        self.part1.gmm_analysis(X_train, X_test, y_train, y_test, 'Wine', 20, 'GMM PCA')
    
    def gmm_pca_nba(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_nba_data_pca_best()
        self.part1.gmm_analysis(X_train, X_test, y_train, y_test, 'NBA', 20, 'GMM PCA')
        
    def gmm_ica_wine(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_wine_data_ica_best()
        self.part1.gmm_analysis(X_train, X_test, y_train, y_test, 'Wine', 20, 'GMM ICA')
    
    def gmm_ica_nba(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_nba_data_ica_best()
        self.part1.gmm_analysis(X_train, X_test, y_train, y_test, 'NBA', 20, 'GMM ICA')
        
    def gmm_rp_wine(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_wine_data_rp_best()
        self.part1.gmm_analysis(X_train, X_test, y_train, y_test, 'Wine', 20, 'GMM RP')
    
    def gmm_rp_nba(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_nba_data_rp_best()
        self.part1.gmm_analysis(X_train, X_test, y_train, y_test, 'NBA', 20, 'GMM RP')
        
    def gmm_lda_wine(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_wine_data_lda_best()
        self.part1.gmm_analysis(X_train, X_test, y_train, y_test, 'Wine', 20, 'GMM LDA')
    
    def gmm_lda_nba(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_nba_data_lda_best()
        self.part1.gmm_analysis(X_train, X_test, y_train, y_test, 'NBA', 20, 'GMM LDA')
        
def main():    
    print('Running part 3')
    
    p = part3()
    
    filename = './' + self.save_dir + '/time.txt'
    with open(filename, 'w') as text_file:
        
        t0 = time()
        p.kmeans_pca_wine()
        text_file.write('kmeans_pca_wine: %0.3f seconds' % (time() - t0))
        
        t0 = time()
        p.kmeans_pca_nba()
        text_file.write('kmeans_pca_nba: %0.3f seconds' % (time() - t0))
        
        t0 = time()
        p.kmeans_ica_wine()
        text_file.write('kmeans_ica_wine: %0.3f seconds' % (time() - t0))
        
        t0 = time()
        p.kmeans_ica_nba()
        text_file.write('kmeans_ica_nba: %0.3f seconds' % (time() - t0))
        
        t0 = time()
        p.kmeans_rp_wine()
        text_file.write('kmeans_rp_wine: %0.3f seconds' % (time() - t0))
        
        t0 = time()
        p.kmeans_rp_nba()
        text_file.write('kmeans_rp_nba: %0.3f seconds' % (time() - t0))
        
        t0 = time()
        p.kmeans_lda_wine()
        text_file.write('kmeans_lda_wine: %0.3f seconds' % (time() - t0))
        
        t0 = time()
        p.kmeans_lda_nba()
        text_file.write('kmeans_lda_nba: %0.3f seconds' % (time() - t0))
        
        t0 = time()
        p.gmm_wine()
        text_file.write('gmm_wine: %0.3f seconds' % (time() - t0))
        
        t0 = time()
        p.gmm_nba()
        text_file.write('gmm_nba: %0.3f seconds' % (time() - t0))
        
        t0 = time()
        p.wine_cluster_plots()
        text_file.write('wine_cluster_plots: %0.3f seconds' % (time() - t0))
        
        t0 = time()
        p.nba_cluster_plots()
        text_file.write('nba_cluster_plots: %0.3f seconds' % (time() - t0))
    
if __name__== '__main__':
    main()
    
