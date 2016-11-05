import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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
#import plot_validation_curve, plot_series

'''
1. Run the clustering algorithms on the data sets and describe what you see.
'''
class part1():
    def __init__(self):
        self.out_dir = 'output_part1'
    
    def cluster_wine(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_wine_data()
        self.kmeans_analysis(X_train, X_test, y_train, y_test, 'Wine', 20)
    
    def cluster_nba(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_nba_data()
        self.kmeans_analysis(X_train, X_test, y_train, y_test, 'NBA', 20)
    
        
    def kmeans_analysis(self, X_train, X_test, y_train, y_test, data_set_name, max_clusters):
        scl = RobustScaler()
        X_train_scl = scl.fit_transform(X_train)
        X_test_scl = scl.transform(X_test)
        
        km_inertias = []
        km_completeness_score = []
        km_homogeneity_score = []
        km_measure_score = []
        km_adjusted_rand_score = []
        km_adjusted_mutual_info_score = []
        km_silhouette_score = []
        
        cluster_range = np.arange(2, max_clusters+1, 1)
        for k in cluster_range:
            print('K Clusters: ', k)
            ##
            ## KMeans
            ##
            km = KMeans(n_clusters=k, algorithm='full', n_jobs=-1)
            km.fit(X_train_scl)
            
            # inertia is the sum of distances from each point to its center   
            km_inertias.append(km.inertia_)
            
            # metrics
            y_train_score = y_train.reshape(y_train.shape[0],)
            
            km_homogeneity_score.append(homogeneity_score(y_train_score, km.labels_))
            km_completeness_score.append(completeness_score(y_train_score, km.labels_))
            km_measure_score.append(v_measure_score(y_train_score, km.labels_))
            km_adjusted_rand_score.append(adjusted_rand_score(y_train_score, km.labels_))
            km_adjusted_mutual_info_score.append(adjusted_mutual_info_score(y_train_score, km.labels_))
            km_silhouette_score.append(silhouette_score(X_train_scl, km.labels_, metric='euclidean'))
            
            
        ##
        ## Elbow Plot
        ##
        title = 'Elbow Plot (K-Means) for ' + data_set_name
        name = data_set_name.lower() + '_kmean_elbow'
        filename = './' + self.out_dir + '/' + name + '.png'
        
        # line to help visualize the elbow
        ph = plot_helper()
        lin = ph.extended_line_from_first_two_points(km_inertias)
        
        ph.plot_series(cluster_range,
                    [km_inertias, lin],
                    [None, None],
                    ['inertia', 'projected'],
                    ['red', 'orange'],
                    ['o', ''],
                    title,
                    'Number of Clusters',
                    'Inertia',
                    filename)
        
        ##
        ## Score Plot
        ##
        title = 'Score Summary Plot (K-Means) for ' + data_set_name
        name = data_set_name.lower() + '_kmean_score'
        filename = './' + self.out_dir + '/' + name + '.png'
                    
        ph.plot_series(cluster_range,
                    [km_homogeneity_score, km_completeness_score, km_measure_score, km_adjusted_rand_score, km_adjusted_mutual_info_score, km_silhouette_score],
                    [None, None, None, None, None, None, None],
                    ['homogeneity', 'completeness', 'measure', 'adjusted_rand', 'adjusted_mutual_info', 'silhouette_score'],
                    ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'],
                    ['o', '^', 'v', '>', '<', '1', '2'],
                    title,
                    'Number of Clusters',
                    'Score',
                    filename)
        
        
    def gmm_analysis(self, X_train, X_test, y_train, y_test, data_set_name, max_clusters):
        scl = RobustScaler()
        X_train_scl = scl.fit_transform(X_train)
        X_test_scl = scl.transform(X_test)
        
        em_bic = []
        em_aic = []
        
        cluster_range = np.arange(2, max_clusters+1, 1)
        for k in cluster_range:
            print('K Clusters: ', k)
            
            ##
            ## Expectation Maximization
            ##
            em = GaussianMixture(n_components=k, covariance_type='full', n_jobs=-1)
            em.fit(X_train_scl)
            em_pred = em.predict(X_train_scl)
            
            em_bic.append(em.bic(X_train_scl))
            em_aic.append(em.aic(X_train_scl))        
        
        ##
        ## Elbow Plot
        ##
        title = 'Elbow Plot (K-Means) for ' + data_set_name
        name = data_set_name.lower() + '_kmean_elbow'
        filename = './' + self.out_dir + '/' + name + '.png'
        
        # line to help visualize the elbow
        ph = plot_helper()
        lin = ph.extended_line_from_first_two_points(km_inertias)
        
        ph.plot_series(cluster_range,
                    [km_inertias, lin],
                    [None, None],
                    ['inertia', 'projected'],
                    ['red', 'orange'],
                    ['o', ''],
                    title,
                    'Number of Clusters',
                    'Inertia',
                    filename)
        
def main():    
    print('Running part 1')
    p = part1()
    p.cluster_wine()
    p.cluster_nba()

if __name__== '__main__':
    main()
    
