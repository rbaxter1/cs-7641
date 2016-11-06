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
        self.out_dir = 'output_part3'
        self.part1 = part1()
    
    def kmeans_pca_wine(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_wine_data()
        self.part1.gmm_analysis(X_train, X_test, y_train, y_test, 'Wine', 20, 'GMM PCA')
    
    def kmeans_pca_nba(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_nba_data()
        self.part1.gmm_analysis(X_train, X_test, y_train, y_test, 'NBA', 20, 'GMM PCA')
        
    def kmeans_ica_wine(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_wine_data()
        self.part1.gmm_analysis(X_train, X_test, y_train, y_test, 'Wine', 20, 'GMM ICA')
    
    def kmeans_ica_nba(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_nba_data()
        self.part1.gmm_analysis(X_train, X_test, y_train, y_test, 'NBA', 20, 'GMM ICA')
        
    def kmeans_rp_wine(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_wine_data()
        self.part1.gmm_analysis(X_train, X_test, y_train, y_test, 'Wine', 20, 'GMM RP')
    
    def kmeans_rp_nba(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_nba_data()
        self.part1.gmm_analysis(X_train, X_test, y_train, y_test, 'NBA', 20, 'GMM RP')
        
    def kmeans_lda_wine(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_wine_data()
        self.part1.gmm_analysis(X_train, X_test, y_train, y_test, 'Wine', 20, 'GMM LDA')
    
    def kmeans_lda_nba(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_nba_data()
        self.part1.gmm_analysis(X_train, X_test, y_train, y_test, 'NBA', 20, 'GMM LDA')
        
    def gmm_pca_wine(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_wine_data()
        self.part1.gmm_analysis(X_train, X_test, y_train, y_test, 'Wine', 20, 'GMM PCA')
    
    def gmm_pca_nba(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_nba_data()
        self.part1.gmm_analysis(X_train, X_test, y_train, y_test, 'NBA', 20, 'GMM PCA')
        
    def gmm_ica_wine(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_wine_data()
        self.part1.gmm_analysis(X_train, X_test, y_train, y_test, 'Wine', 20, 'GMM ICA')
    
    def gmm_ica_nba(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_nba_data()
        self.part1.gmm_analysis(X_train, X_test, y_train, y_test, 'NBA', 20, 'GMM ICA')
        
    def gmm_rp_wine(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_wine_data()
        self.part1.gmm_analysis(X_train, X_test, y_train, y_test, 'Wine', 20, 'GMM RP')
    
    def gmm_rp_nba(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_nba_data()
        self.part1.gmm_analysis(X_train, X_test, y_train, y_test, 'NBA', 20, 'GMM RP')
        
    def gmm_lda_wine(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_wine_data()
        self.part1.gmm_analysis(X_train, X_test, y_train, y_test, 'Wine', 20, 'GMM LDA')
    
    def gmm_lda_nba(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_nba_data()
        self.part1.gmm_analysis(X_train, X_test, y_train, y_test, 'NBA', 20, 'GMM LDA')
        




        
        
    def wine_cluster_plots(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_wine_data()
        
        x_col_names = ['Alcohol', 'Volatile Acidity', 'Sulphates', 'pH']
        df = pd.DataFrame(X_train)
        df.columns = x_col_names
        
        self.cluster_plot(df, 5, 'KMeans', 'Wine')
        self.cluster_plot(df, 3, 'GaussianMixture', 'Wine')
        self.cluster_3d_plot(df, 5, 'KMeans', 'Wine')
        self.cluster_3d_plot(df, 3, 'GaussianMixture', 'Wine')
        
    def nba_cluster_plots(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_nba_data()
        
        x_col_names = ['Shot Distance', 'Closest Defender Distance', 'Number Dribbles']
        df = pd.DataFrame(X_train)
        df.columns = x_col_names
        
        self.cluster_plot(df, 2, 'KMeans', 'NBA')
        self.cluster_plot(df, 8, 'GaussianMixture', 'NBA')
        self.cluster_3d_plot(df, 2, 'KMeans', 'NBA')
        self.cluster_3d_plot(df, 8, 'GaussianMixture', 'NBA')
    
    def gmm_wine(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_wine_data()
        self.gmm_analysis(X_train, X_test, y_train, y_test, 'Wine', 30)
    
    def gmm_nba(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_nba_data()
        self.gmm_analysis(X_train, X_test, y_train, y_test, 'NBA', 30)
        
    
    
        
def main():    
    print('Running part 3')
    
    p = part1()
    
    t0 = time()
    
    p.kmeans_pca_wine()
    p.kmeans_pca_nba()
    
    p.kmeans_ica_wine()
    p.kmeans_ica_nba()
    
    p.kmeans_rp_wine()
    p.kmeans_rp_nba()
    
    p.kmeans_lda_wine()
    p.kmeans_lda_nba()
    
    
    p.gmm_wine()
    p.gmm_nba()
    
    p.wine_cluster_plots()
    p.nba_cluster_plots()
    
    print("done in %0.3f seconds" % (time() - t0))
    
if __name__== '__main__':
    main()
    
