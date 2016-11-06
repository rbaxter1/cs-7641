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

'''
1. Run the clustering algorithms on the data sets and describe what you see.
'''
class part1():
    def __init__(self):
        self.out_dir = 'output_part1'
    
    def run(self):
        print('Running part 1')
    
        filename = './' + self.out_dir + '/time.txt'
        with open(filename, 'w') as text_file:
            
            t0 = time()
            self.wine_cluster_plots()
            text_file.write('wine_cluster_plots: %0.3f seconds' % (time() - t0))
            
            t0 = time()
            self.nba_cluster_plots()
            text_file.write('nba_cluster_plots: %0.3f seconds' % (time() - t0))
            
            t0 = time()
            self.gmm_wine()
            text_file.write('gmm_wine: %0.3f seconds' % (time() - t0))
            
            t0 = time()
            self.gmm_nba()
            text_file.write('gmm_nba: %0.3f seconds' % (time() - t0))
            
            t0 = time()
            self.kmeans_wine()
            text_file.write('kmeans_wine: %0.3f seconds' % (time() - t0))
            
            t0 = time()
            self.kmeans_nba()
            text_file.write('kmeans_nba: %0.3f seconds' % (time() - t0))
            
            
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
        
    def kmeans_wine(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_wine_data()
        self.kmeans_analysis(X_train, X_test, y_train, y_test, 'Wine', 20)
    
    def kmeans_nba(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_nba_data()
        self.kmeans_analysis(X_train, X_test, y_train, y_test, 'NBA', 20)
    
    # source: https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch11/ch11.ipynb
    def silhouette_plot(self, X, X_predicted, title, filename):
        plt.clf()
        plt.cla()
        
        cluster_labels = np.unique(X_predicted)
        n_clusters = cluster_labels.shape[0]
        silhouette_vals = silhouette_samples(X, X_predicted, metric='euclidean')
        y_ax_lower, y_ax_upper = 0, 0
        
        color=iter(cm.viridis(np.linspace(0,1,cluster_labels.shape[0])))
           
        yticks = []
        for i, c in enumerate(cluster_labels):
            c_silhouette_vals = silhouette_vals[X_predicted == c]
            c_silhouette_vals.sort()
            y_ax_upper += len(c_silhouette_vals)
            plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, edgecolor='none', color=next(color))
        
            yticks.append((y_ax_lower + y_ax_upper) / 2.)
            y_ax_lower += len(c_silhouette_vals)
            
        silhouette_avg = np.mean(silhouette_vals)
        plt.axvline(silhouette_avg, color="red", linestyle="--") 
        
        plt.yticks(yticks, cluster_labels + 1)
        plt.ylabel('Cluster')
        plt.xlabel('Silhouette Coefficient')
        
        plt.title(title)
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close('all')
        
    def cluster_plot(self, df, k, cls_type, data_set_name):
        for i in range(df.shape[1]):
            for j in range(df.shape[1]):
        
                if i == j:
                    continue
            
                f1 = df.columns[i]
                f2 = df.columns[j]
                
                print('Feature1: ', f1, ', Feature2: ', f2)
                X_train = df.values[:,(i,j)]
                X_train_scl = RobustScaler().fit_transform(X_train)
                
                ##
                ## Cluster Routine
                ##
                if 'KMeans' in cls_type:
                    cls = KMeans(n_clusters=k, algorithm='full')
                elif 'GaussianMixture' in cls_type:
                    cls = GaussianMixture(n_components=k, covariance_type='full')
                else:
                    raise AttributeError('cls_type: ' + cls_type + ' not supported.')
                                         
                cls.fit(X_train_scl)
                y_pred = cls.predict(X_train_scl)
                
                # Clusters plot
                plt.clf()
                plt.cla()
                plt.scatter(X_train_scl[:,0], X_train_scl[:,1], c=y_pred)
                
                title = cls_type + ' Clusters: ' + f1 + ' vs ' + f2 + ', k=' + str(k) + ' (' + data_set_name + ')'
                plt.title(title)
                plt.xlabel(f1)
                plt.ylabel(f2)
                
                fn = './' + self.out_dir + '/' + f1.lower() + '_' + f2.lower() + '_' + str(k) + '_' + cls_type.lower() + '_' + data_set_name + '_cluster.png'
                plt.savefig(fn)
                plt.close('all')
                                
    
    def cluster_3d_plot(self, df, k, cls_type, data_set_name):
        p = list(itertools.permutations(range(df.shape[1]), 3))

        print(p)
        
        for u in p:            
            f1 = df.columns[u[0]]
            f2 = df.columns[u[1]]
            f3 = df.columns[u[2]]
            
            print('Feature1: ', f1, ', Feature2: ', f2, ', Feature3: ', f3)
            X_train = df.values[:,(u[0],u[1],u[2])]
            X_train_scl = RobustScaler().fit_transform(X_train)
            
            ##
            ## Cluster Routine
            ##
            if 'KMeans' in cls_type:
                cls = KMeans(n_clusters=k, algorithm='full')
            elif 'GaussianMixture' in cls_type:
                cls = GaussianMixture(n_components=k, covariance_type='full')
            else:
                raise AttributeError('cls_type: ' + cls_type + ' not supported.')
                                     
            cls.fit(X_train_scl)
            y_pred = cls.predict(X_train_scl)
            
            ##
            ## Plots
            ##
            ph = plot_helper()
            
            ##
            ## 3d Scatter Plot
            ##
            title = cls_type + ' Clusters 3D: ' + f1 + '\nvs ' + f2 + ' vs ' + f3 + ', k=' + str(k)
            name = data_set_name.lower() + '_' + cls_type.lower() + '3d_cluster'
            filename = './' + self.out_dir + '/' + f1.lower() + '_' + f2.lower() + '_' + f3.lower() + '_' + str(k) + '_' + cls_type.lower() + '_' + data_set_name + '_cluster.png'
            
            ph.plot_3d_scatter(X_train_scl[:,0], X_train_scl[:,1], X_train_scl[:,2], y_pred, f1, f2, f3, title, filename)
            
                    
    def kmeans_analysis(self, X_train, X_test, y_train, y_test, data_set_name, max_clusters, analysis_name='K-Means'):
        scl = RobustScaler()
        X_train_scl = scl.fit_transform(X_train)
        X_test_scl = scl.transform(X_test)
        
        km_inertias = []
        km_completeness_score = []
        km_homogeneity_score = []
        km_measure_score = []
        km_adjusted_rand_score = []
        km_adjusted_mutual_info_score = []
        
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
            
            ##
            ## Silhouette Plot
            ##
            title = 'Silhouette Plot (' + analysis_name + ', k=' + str(k) + ') for ' + data_set_name
            name = data_set_name.lower() + '_' + analysis_name.lower() + '_silhouette_' + str(k)
            filename = './' + self.out_dir + '/' + name + '.png'
            
            self.silhouette_plot(X_train_scl, km.labels_, title, filename)
            
        ##
        ## Plots
        ##
        ph = plot_helper()
        
        ##
        ## Elbow Plot
        ##
        title = 'Elbow Plot (' + analysis_name + ') for ' + data_set_name
        name = data_set_name.lower() + '_' + analysis_name.lower() + '_elbow'
        filename = './' + self.out_dir + '/' + name + '.png'
        
        # line to help visualize the elbow
        lin = ph.extended_line_from_first_two_points(km_inertias, 0, 2)
        
        ph.plot_series(cluster_range,
                    [km_inertias, lin],
                    [None, None],
                    ['inertia', 'projected'],
                    cm.viridis(np.linspace(0, 1, 2)),
                    ['o', ''],
                    title,
                    'Number of Clusters',
                    'Inertia',
                    filename)
        
        ##
        ## Score Plot
        ##
        title = 'Score Summary Plot (' + analysis_name + ') for ' + data_set_name
        name = data_set_name.lower() + '_' + analysis_name.lower() + '_score'
        filename = './' + self.out_dir + '/' + name + '.png'
                    
        ph.plot_series(cluster_range,
                    [km_homogeneity_score, km_completeness_score, km_measure_score, km_adjusted_rand_score, km_adjusted_mutual_info_score],
                    [None, None, None, None, None, None],
                    ['homogeneity', 'completeness', 'measure', 'adjusted_rand', 'adjusted_mutual_info'],
                    cm.viridis(np.linspace(0, 1, 5)),
                    ['o', '^', 'v', '>', '<', '1'],
                    title,
                    'Number of Clusters',
                    'Score',
                    filename)
        
    def gmm_analysis(self, X_train, X_test, y_train, y_test, data_set_name, max_clusters, analysis_name='GMM'):
        scl = RobustScaler()
        X_train_scl = scl.fit_transform(X_train)
        X_test_scl = scl.transform(X_test)
        
        em_bic = []
        em_aic = []
        em_completeness_score = []
        em_homogeneity_score = []
        em_measure_score = []
        em_adjusted_rand_score = []
        em_adjusted_mutual_info_score = []
        
        cluster_range = np.arange(2, max_clusters+1, 1)
        for k in cluster_range:
            print('K Clusters: ', k)
            
            ##
            ## Expectation Maximization
            ##
            em = GaussianMixture(n_components=k, covariance_type='full')
            em.fit(X_train_scl)
            em_pred = em.predict(X_train_scl)
            
            em_bic.append(em.bic(X_train_scl))
            em_aic.append(em.aic(X_train_scl))        
        
            # metrics
            y_train_score = y_train.reshape(y_train.shape[0],)
            
            em_homogeneity_score.append(homogeneity_score(y_train_score, em_pred))
            em_completeness_score.append(completeness_score(y_train_score, em_pred))
            em_measure_score.append(v_measure_score(y_train_score, em_pred))
            em_adjusted_rand_score.append(adjusted_rand_score(y_train_score, em_pred))
            em_adjusted_mutual_info_score.append(adjusted_mutual_info_score(y_train_score, em_pred))
            
        
        ##
        ## Plots
        ##
        ph = plot_helper()
        
        ##
        ## BIC/AIC Plot
        ##
        title = 'Information Criterion Plot (' + analysis_name + ') for ' + data_set_name
        name = data_set_name.lower() + '_' + analysis_name.lower() + '_ic'
        filename = './' + self.out_dir + '/' + name + '.png'
        
        ph.plot_series(cluster_range,
                    [em_bic, em_aic],
                    [None, None],
                    ['bic', 'aic'],
                    cm.viridis(np.linspace(0, 1, 2)),
                    ['o', '*'],
                    title,
                    'Number of Clusters',
                    'Information Criterion',
                    filename)
        
        ##
        ## Score Plot
        ##
        title = 'Score Summary Plot (' + analysis_name + ') for ' + data_set_name
        name = data_set_name.lower() + '_' + analysis_name.lower() + '_score'
        filename = './' + self.out_dir + '/' + name + '.png'
                    
        ph.plot_series(cluster_range,
                    [em_homogeneity_score, em_completeness_score, em_measure_score, em_adjusted_rand_score, em_adjusted_mutual_info_score],
                    [None, None, None, None, None, None],
                    ['homogeneity', 'completeness', 'measure', 'adjusted_rand', 'adjusted_mutual_info'],
                    cm.viridis(np.linspace(0, 1, 5)),
                    ['o', '^', 'v', '>', '<', '1'],
                    title,
                    'Number of Clusters',
                    'Score',
                    filename)
        
def main():
    p = part1()
    p.run()
    
    
if __name__== '__main__':
    main()
    
