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
        self.save_dir = 'data'
        self.out_dir = 'output_part3'
        self.part1 = part1()
        self.part1.out_dir = self.out_dir
        self.time_filename = './' + self.out_dir + '/time.txt'
    
    def run(self):
        print('Running part 3')
        with open(self.time_filename, 'w') as text_file:
            '''
            t0 = time()
            self.kmeans_pca_wine()
            text_file.write('kmeans_pca_wine: %0.3f seconds\n' % (time() - t0))
            
            t0 = time()
            self.kmeans_pca_nba()
            text_file.write('kmeans_pca_nba: %0.3f seconds\n' % (time() - t0))
            
            t0 = time()
            self.kmeans_ica_wine()
            text_file.write('kmeans_ica_wine: %0.3f seconds\n' % (time() - t0))
            
            t0 = time()
            self.kmeans_ica_nba()
            text_file.write('kmeans_ica_nba: %0.3f seconds\n' % (time() - t0))
            
            t0 = time()
            self.kmeans_rp_wine()
            text_file.write('kmeans_rp_wine: %0.3f seconds\n' % (time() - t0))
            
            t0 = time()
            self.kmeans_rp_nba()
            text_file.write('kmeans_rp_nba: %0.3f seconds\n' % (time() - t0))
            
            t0 = time()
            self.kmeans_lda_wine()
            text_file.write('kmeans_lda_wine: %0.3f seconds\n' % (time() - t0))
            
            t0 = time()
            self.kmeans_lda_nba()
            text_file.write('kmeans_lda_nba: %0.3f seconds\n' % (time() - t0))
            
            t0 = time()
            self.gmm_pca_wine()
            text_file.write('gmm_wine: %0.3f seconds\n' % (time() - t0))
            
            t0 = time()
            self.gmm_pca_nba()
            text_file.write('gmm_nba: %0.3f seconds\n' % (time() - t0))
            
            t0 = time()
            self.gmm_ica_wine()
            text_file.write('gmm_wine: %0.3f seconds\n' % (time() - t0))
            
            t0 = time()
            self.gmm_ica_nba()
            text_file.write('gmm_nba: %0.3f seconds\n' % (time() - t0))
            
            t0 = time()
            self.gmm_rp_wine()
            text_file.write('gmm_wine: %0.3f seconds\n' % (time() - t0))
            
            t0 = time()
            self.gmm_rp_nba()
            text_file.write('gmm_nba: %0.3f seconds\n' % (time() - t0))
            
            t0 = time()
            self.gmm_lda_wine()
            text_file.write('gmm_wine: %0.3f seconds\n' % (time() - t0))
            
            t0 = time()
            self.gmm_lda_nba()
            text_file.write('gmm_nba: %0.3f seconds\n' % (time() - t0))
            '''
            t0 = time()
            self.wine_cluster_plots()
            text_file.write('wine_cluster_plots: %0.3f seconds\n' % (time() - t0))
            
            t0 = time()
            self.nba_cluster_plots()
            text_file.write('nba_cluster_plots: %0.3f seconds\n' % (time() - t0))
            
            ##
            ## Generate files for best
            ##
            
            # only need wine for parts 4 and 5
            self.best_pca_cluster_wine()
            self.best_ica_cluster_wine()
            self.best_rp_cluster_wine()
            self.best_lda_cluster_wine()
            
            

    def best_pca_cluster_wine(self):
        dh = data_helper()
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_wine_data_pca_best()
        
        scl = RobustScaler()
        X_train_scl = scl.fit_transform(X_train)
        X_test_scl = scl.transform(X_test)
        
        ##
        ## K-Means
        ##
        km = KMeans(n_clusters=3, algorithm='full')
        X_train_transformed = km.fit_transform(X_train_scl)
        X_test_transformed = km.transform(X_test_scl)
        
        # save
        filename = './' + self.save_dir + '/wine_kmeans_pca_x_train.txt'
        pd.DataFrame(X_train_transformed).to_csv(filename, header=False, index=False)
        
        filename = './' + self.save_dir + '/wine_kmeans_pca_x_test.txt'
        pd.DataFrame(X_test_transformed).to_csv(filename, header=False, index=False)
        
        filename = './' + self.save_dir + '/wine_kmeans_pca_y_train.txt'
        pd.DataFrame(y_train).to_csv(filename, header=False, index=False)
        
        filename = './' + self.save_dir + '/wine_kmeans_pca_y_test.txt'
        pd.DataFrame(y_test).to_csv(filename, header=False, index=False)
        
        ##
        ## GMM
        ##
        gmm = GaussianMixture(n_components=3, covariance_type='full')
        X_train_transformed = km.fit_transform(X_train_scl)
        X_test_transformed = km.transform(X_test_scl)
        
        # save
        filename = './' + self.save_dir + '/wine_gmm_pca_x_train.txt'
        pd.DataFrame(X_train_transformed).to_csv(filename, header=False, index=False)
        
        filename = './' + self.save_dir + '/wine_gmm_pca_x_test.txt'
        pd.DataFrame(X_test_transformed).to_csv(filename, header=False, index=False)
        
        filename = './' + self.save_dir + '/wine_gmm_pca_y_train.txt'
        pd.DataFrame(y_train).to_csv(filename, header=False, index=False)
        
        filename = './' + self.save_dir + '/wine_gmm_pca_y_test.txt'
        pd.DataFrame(y_test).to_csv(filename, header=False, index=False)
        
    def best_ica_cluster_wine(self):
        dh = data_helper()
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_wine_data_ica_best()
        
        scl = RobustScaler()
        X_train_scl = scl.fit_transform(X_train)
        X_test_scl = scl.transform(X_test)
        
        ##
        ## K-Means
        ##
        km = KMeans(n_clusters=3, algorithm='full')
        X_train_transformed = km.fit_transform(X_train_scl)
        X_test_transformed = km.transform(X_test_scl)
        
        # save
        filename = './' + self.save_dir + '/wine_kmeans_ica_x_train.txt'
        pd.DataFrame(X_train_transformed).to_csv(filename, header=False, index=False)
        
        filename = './' + self.save_dir + '/wine_kmeans_ica_x_test.txt'
        pd.DataFrame(X_test_transformed).to_csv(filename, header=False, index=False)
        
        filename = './' + self.save_dir + '/wine_kmeans_ica_y_train.txt'
        pd.DataFrame(y_train).to_csv(filename, header=False, index=False)
        
        filename = './' + self.save_dir + '/wine_kmeans_ica_y_test.txt'
        pd.DataFrame(y_test).to_csv(filename, header=False, index=False)
        
        ##
        ## GMM
        ##
        gmm = GaussianMixture(n_components=4, covariance_type='full')
        X_train_transformed = km.fit_transform(X_train_scl)
        X_test_transformed = km.transform(X_test_scl)
        
        # save
        filename = './' + self.save_dir + '/wine_gmm_ica_x_train.txt'
        pd.DataFrame(X_train_transformed).to_csv(filename, header=False, index=False)
        
        filename = './' + self.save_dir + '/wine_gmm_ica_x_test.txt'
        pd.DataFrame(X_test_transformed).to_csv(filename, header=False, index=False)
        
        filename = './' + self.save_dir + '/wine_gmm_ica_y_train.txt'
        pd.DataFrame(y_train).to_csv(filename, header=False, index=False)
        
        filename = './' + self.save_dir + '/wine_gmm_ica_y_test.txt'
        pd.DataFrame(y_test).to_csv(filename, header=False, index=False)
        
    def best_rp_cluster_wine(self):
        dh = data_helper()
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_wine_data_rp_best()
        
        scl = RobustScaler()
        X_train_scl = scl.fit_transform(X_train)
        X_test_scl = scl.transform(X_test)
        
        ##
        ## K-Means
        ##
        km = KMeans(n_clusters=5, algorithm='full')
        X_train_transformed = km.fit_transform(X_train_scl)
        X_test_transformed = km.transform(X_test_scl)
        
        # save
        filename = './' + self.save_dir + '/wine_kmeans_rp_x_train.txt'
        pd.DataFrame(X_train_transformed).to_csv(filename, header=False, index=False)
        
        filename = './' + self.save_dir + '/wine_kmeans_rp_x_test.txt'
        pd.DataFrame(X_test_transformed).to_csv(filename, header=False, index=False)
        
        filename = './' + self.save_dir + '/wine_kmeans_rp_y_train.txt'
        pd.DataFrame(y_train).to_csv(filename, header=False, index=False)
        
        filename = './' + self.save_dir + '/wine_kmeans_rp_y_test.txt'
        pd.DataFrame(y_test).to_csv(filename, header=False, index=False)
        
        ##
        ## GMM
        ##
        gmm = GaussianMixture(n_components=3, covariance_type='full')
        X_train_transformed = km.fit_transform(X_train_scl)
        X_test_transformed = km.transform(X_test_scl)
        
        # save
        filename = './' + self.save_dir + '/wine_gmm_rp_x_train.txt'
        pd.DataFrame(X_train_transformed).to_csv(filename, header=False, index=False)
        
        filename = './' + self.save_dir + '/wine_gmm_rp_x_test.txt'
        pd.DataFrame(X_test_transformed).to_csv(filename, header=False, index=False)
        
        filename = './' + self.save_dir + '/wine_gmm_rp_y_train.txt'
        pd.DataFrame(y_train).to_csv(filename, header=False, index=False)
        
        filename = './' + self.save_dir + '/wine_gmm_rp_y_test.txt'
        pd.DataFrame(y_test).to_csv(filename, header=False, index=False)
    
    def best_lda_cluster_wine(self):
        dh = data_helper()
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_wine_data_lda_best()
        
        scl = RobustScaler()
        X_train_scl = scl.fit_transform(X_train)
        X_test_scl = scl.transform(X_test)
        
        ##
        ## K-Means
        ##
        km = KMeans(n_clusters=4, algorithm='full')
        X_train_transformed = km.fit_transform(X_train_scl)
        X_test_transformed = km.transform(X_test_scl)
        
        # save
        filename = './' + self.save_dir + '/wine_kmeans_lda_x_train.txt'
        pd.DataFrame(X_train_transformed).to_csv(filename, header=False, index=False)
        
        filename = './' + self.save_dir + '/wine_kmeans_lda_x_test.txt'
        pd.DataFrame(X_test_transformed).to_csv(filename, header=False, index=False)
        
        filename = './' + self.save_dir + '/wine_kmeans_lda_y_train.txt'
        pd.DataFrame(y_train).to_csv(filename, header=False, index=False)
        
        filename = './' + self.save_dir + '/wine_kmeans_lda_y_test.txt'
        pd.DataFrame(y_test).to_csv(filename, header=False, index=False)
        
        ##
        ## GMM
        ##
        gmm = GaussianMixture(n_components=4, covariance_type='full')
        X_train_transformed = km.fit_transform(X_train_scl)
        X_test_transformed = km.transform(X_test_scl)
        
        # save
        filename = './' + self.save_dir + '/wine_gmm_lda_x_train.txt'
        pd.DataFrame(X_train_transformed).to_csv(filename, header=False, index=False)
        
        filename = './' + self.save_dir + '/wine_gmm_lda_x_test.txt'
        pd.DataFrame(X_test_transformed).to_csv(filename, header=False, index=False)
        
        filename = './' + self.save_dir + '/wine_gmm_lda_y_train.txt'
        pd.DataFrame(y_train).to_csv(filename, header=False, index=False)
        
        filename = './' + self.save_dir + '/wine_gmm_lda_y_test.txt'
        pd.DataFrame(y_test).to_csv(filename, header=False, index=False)
        
        
    def wine_cluster_plots(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_wine_data_pca_best()
        
        df = pd.DataFrame(X_train)
        
        self.part1.cluster_plot(df, 3, 'KMeans', 'Wine', 'K-Means PCA')
        self.part1.cluster_plot(df, 3, 'GaussianMixture', 'Wine', 'GMM PCA')
        if df.shape[1] >=3:
            self.part1.cluster_3d_plot(df, 3, 'KMeans', 'Wine', 'K-Means PCA')
            self.part1.cluster_3d_plot(df, 3, 'GaussianMixture', 'Wine', 'GMM PCA')
        
        
        X_train, X_test, y_train, y_test = dh.get_wine_data_ica_best()
        
        df = pd.DataFrame(X_train)
        
        self.part1.cluster_plot(df, 3, 'KMeans', 'Wine', 'K-Means ICA')
        self.part1.cluster_plot(df, 4, 'GaussianMixture', 'Wine', 'GMM ICA')
        if df.shape[1] >=3:
            self.part1.cluster_3d_plot(df, 3, 'KMeans', 'Wine', 'K-Means ICA')
            self.part1.cluster_3d_plot(df, 4, 'GaussianMixture', 'Wine', 'GMM ICA')
        
        
        X_train, X_test, y_train, y_test = dh.get_wine_data_lda_best()
        
        df = pd.DataFrame(X_train)
        
        self.part1.cluster_plot(df, 4, 'KMeans', 'Wine', 'K-Means LDA')
        self.part1.cluster_plot(df, 4, 'GaussianMixture', 'Wine', 'GMM LDA')
        if df.shape[1] >=3:
            self.part1.cluster_3d_plot(df, 4, 'KMeans', 'Wine', 'K-Means LDA')
            self.part1.cluster_3d_plot(df, 4, 'GaussianMixture', 'Wine', 'GMM LDA')
        
        
        X_train, X_test, y_train, y_test = dh.get_wine_data_rp_best()
        
        df = pd.DataFrame(X_train)
        
        self.part1.cluster_plot(df, 5, 'KMeans', 'Wine', 'K-Means RP')
        self.part1.cluster_plot(df, 3, 'GaussianMixture', 'Wine', 'GMM RP')
        if df.shape[1] >=3:
            self.part1.cluster_3d_plot(df, 5, 'KMeans', 'Wine', 'K-Means RP')
            self.part1.cluster_3d_plot(df, 3, 'GaussianMixture', 'Wine', 'GMM RP')
        
        
    def nba_cluster_plots(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_nba_data_pca_best()
        
        df = pd.DataFrame(X_train)
        
        self.part1.cluster_plot(df, 3, 'KMeans', 'NBA', 'K-Means PCA')
        self.part1.cluster_plot(df, 8, 'GaussianMixture', 'NBA', 'GMM PCA')
        if df.shape[1] >=3:
            self.part1.cluster_3d_plot(df, 3, 'KMeans', 'NBA', 'K-Means PCA')
            self.part1.cluster_3d_plot(df, 8, 'GaussianMixture', 'NBA', 'GMM PCA')
        
        
        X_train, X_test, y_train, y_test = dh.get_nba_data_ica_best()
        
        df = pd.DataFrame(X_train)
        
        self.part1.cluster_plot(df, 3, 'KMeans', 'NBA', 'K-Means ICA')
        self.part1.cluster_plot(df, 5, 'GaussianMixture', 'NBA', 'GMM ICA')
        if df.shape[1] >=3:
            self.part1.cluster_3d_plot(df, 3, 'KMeans', 'NBA', 'K-Means ICA')
            self.part1.cluster_3d_plot(df, 5, 'GaussianMixture', 'NBA', 'GMM ICA')
        
        
        X_train, X_test, y_train, y_test = dh.get_nba_data_lda_best()
        
        df = pd.DataFrame(X_train)
        
        self.part1.cluster_plot(df, 4, 'KMeans', 'NBA', 'K-Means LDA')
        self.part1.cluster_plot(df, 3, 'GaussianMixture', 'NBA', 'GMM LDA')
        if df.shape[1] >=3:
            self.part1.cluster_3d_plot(df, 4, 'KMeans', 'NBA', 'K-Means LDA')
            self.part1.cluster_3d_plot(df, 3, 'GaussianMixture', 'NBA', 'GMM LDA')
        
        
        X_train, X_test, y_train, y_test = dh.get_nba_data_rp_best()
        
        df = pd.DataFrame(X_train)
        
        self.part1.cluster_plot(df, 4, 'KMeans', 'NBA', 'K-Means RP')
        self.part1.cluster_plot(df, 5, 'GaussianMixture', 'NBA', 'GMM RP')
        if df.shape[1] >=3:
            self.part1.cluster_3d_plot(df, 4, 'KMeans', 'NBA', 'K-Means RP')
            self.part1.cluster_3d_plot(df, 5, 'GaussianMixture', 'NBA', 'GMM RP')
        
        
    def kmeans_pca_wine(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_wine_data_pca_best()
        self.part1.kmeans_analysis(X_train, X_test, y_train, y_test, 'Wine', 20, 'K-Means PCA')
    
    def kmeans_pca_nba(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_nba_data_pca_best()
        self.part1.kmeans_analysis(X_train, X_test, y_train, y_test, 'NBA', 20, 'K-Means PCA')
        
    def kmeans_ica_wine(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_wine_data_ica_best()
        self.part1.kmeans_analysis(X_train, X_test, y_train, y_test, 'Wine', 20, 'K-Means ICA')
    
    def kmeans_ica_nba(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_nba_data_ica_best()
        self.part1.kmeans_analysis(X_train, X_test, y_train, y_test, 'NBA', 20, 'K-Means ICA')
        
    def kmeans_rp_wine(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_wine_data_rp_best()
        self.part1.kmeans_analysis(X_train, X_test, y_train, y_test, 'Wine', 20, 'K-Means RP')
    
    def kmeans_rp_nba(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_nba_data_rp_best()
        self.part1.kmeans_analysis(X_train, X_test, y_train, y_test, 'NBA', 20, 'K-Means RP')
        
    def kmeans_lda_wine(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_wine_data_lda_best()
        self.part1.kmeans_analysis(X_train, X_test, y_train, y_test, 'Wine', 20, 'K-Means LDA')
    
    def kmeans_lda_nba(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_nba_data_lda_best()
        self.part1.kmeans_analysis(X_train, X_test, y_train, y_test, 'NBA', 20, 'K-Means LDA')
        
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
    p = part3()
    p.run()
    
if __name__== '__main__':
    main()
    
