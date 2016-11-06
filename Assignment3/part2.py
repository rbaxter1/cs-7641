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
from sklearn.metrics import mean_squared_error

from scipy.stats import kurtosis

from timeit import default_timer as time

from data_helper import *
from plot_nn import plot_validation_curve, plot_series


'''
2. Apply the dimensionality reduction algorithms to the two datasets and describe what you see.
'''        
class part2():
    def __init__(self):
        self.out_dir = 'output_part2'

    def pca_wine(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_wine_data()
        self.gmm_analysis(X_train, X_test, y_train, y_test, 'Wine', 30)
    
    def ica_nba(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_nba_data()
        self.gmm_analysis(X_train, X_test, y_train, y_test, 'NBA', 30)
        
        
    def reconstruction_error(self, X_train_scl, cls):
        rng = range(1, X_train_scl.shape[1]+1)
        
        all_mses = np.ndarray([0, len(rng)])
        for n in range(100):
            mses = []
            for i in rng:
                dr = cls(n_components=i)
                dr.fit(X_train_scl)
                
                X_projected = dr.inverse_transform(dr.labels_)
                mse = mean_squared_error(X_train_scl, X_projected)
                mses.append(mse)
                print(i, mse)
            
            all_mses = np.vstack([all_mses, mses])
            
        return all_mses, rng
            
    def plot_pca_scatter(self, X, y, title, filename, f0_name='feature 1', f1_name='feature 2', x0_i=0, x1_i=1):
        
        y.shape = (y.shape[0],)
        
        plt.clf()
        plt.cla()

        for i in np.unique(y):
            plt.scatter(X[y==i, x0_i], X[y==i, x1_i], label=i, alpha=.5)
        
        plt.title(title)
        plt.xlabel(f0_name)
        plt.ylabel(f1_name)
        
        plt.legend(loc="best")
        
        plt.savefig(filename)
        plt.close('all')

    def plot_pca(self, pca, title, filename):
        
        plt.clf()
        plt.cla()
        fig, ax = plt.subplots()
        
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        
        rng = np.arange(1, pca.explained_variance_ratio_.shape[0]+1, 1)
        
        plt.bar(rng, pca.explained_variance_ratio_,
                alpha=0.5, align='center',
                label='Individual Explained Variance')
        
        plt.step(rng, np.cumsum(pca.explained_variance_ratio_),
                where='mid', label='Cumulative Explained Variance')
        
        plt.legend(loc='best')
        
        plt.title(title)
        
        plt.xlabel('Principal Components')
        plt.ylabel('Explained Variance Ratio')
        
        
        plt.savefig(filename)
        plt.close('all')
        
        
    def pca_wine(self):
        
        ##
        ## Data
        ##
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_wine_data()
        
        scl = RobustScaler()
        X_train_scl = scl.fit_transform(X_train)
        X_test_scl = scl.transform(X_test)
        
        ##
        ## PCA
        ##
        pca = PCA(n_components=X_train_scl.shape[1], svd_solver='full')
        X_pca = pca.fit_transform(X_train_scl)
        
        print(X_pca.shape)
        
        title = 'PCA Explained Variance: Wine'
        filename = './' + self.out_dir + '/wine_pca_evr.png'    
        self.plot_pca(pca, title, filename)

        ##
        ## PCA
        ##
        pca = PCA(n_components=2, svd_solver='full')
        X_pca = pca.fit_transform(X_train_scl)

        title = 'PCA Scatter: Wine'
        filename = './' + self.out_dir + '/wine_pca_sc.png'    
        self.plot_pca_scatter(X_pca, y_train, title, filename, f0_name='feature 1', f1_name='feature 2', x0_i=0, x1_i=1)    
        
        ##
        ## PCA
        ##
        # questionable
        '''
        pipe = Pipeline([('scl', RobustScaler()),
                         ('pca', PCA(svd_solver='full'))])
        
        cv = StratifiedShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
        
        title = "Validation Curves (PCA)"
        name = 'pca_vc_n_components'
        fn = './' + self.out_dir + '/' + name + '.png'
        
        plot_validation_curve(X_train,
                              X_test,
                              y_train,
                              y_test,
                              pipe,
                              'pca__n_components',
                              np.arange(2, X_train.shape[1]+1, 1),
                              title,
                              fn,
                              cv=cv,
                              param_range_plot=None)
        '''
        
        ##
        ## PCA
        ##
        '''
        rng = range(1, X_train.shape[1]+1)
        all_mses = np.ndarray([0, len(rng)])
        for n in range(100):
            mses = []
            for i in rng:
                pca = PCA(n_components=i)
                X_train_pca = pca.fit_transform(X_train_scl)
                
                X_projected = pca.inverse_transform(X_train_pca)
                mse = mean_squared_error(X_train_scl, X_projected)
                mses.append(mse)
                print(i, mse)
            
            all_mses = np.vstack([all_mses, mses])
            
        '''
        
        all_mses, rng = self.reconstruction_error(X_train_scl, PCA)
        
        title = "Reconstruction Error (PCA)"
        name = 'pca_rec_err'
        filename = './' + self.out_dir + '/' + name + '.png'
        
        plot_series(rng,
                    [all_mses.mean(0)],
                    [all_mses.std(0)],
                    ['mean square error'],
                    ['red'],
                    ['o'],
                    title,
                    'Number of Features',
                    'Mean Square Error',
                    filename)
            
        
    def ica_wine(self):
        
        ##
        ## Data
        ##
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
        
        # top 2
        X_ica_top2 = X_ica_sorted[:,0:2]
        
        
        all_mses, rng = self.reconstruction_error(X_ica_top2, FastICA)
        
        title = "Reconstruction Error (FastICA)"
        name = 'ica_rec_err'
        filename = './' + self.out_dir + '/' + name + '.png'
        
        plot_series(rng,
                    [all_mses.mean(0)],
                    [all_mses.std(0)],
                    ['mean square error'],
                    ['red'],
                    ['o'],
                    title,
                    'Number of Features',
                    'Mean Square Error',
                    filename)
        
        
        '''
        #questionable
        title = "Kurtosis (ICA)"
        name = 'ica_kurt'
        filename = './' + self.out_dir + '/' + name + '.png'
        
        plot_series(i,
                    [kurt],
                    [None],
                    ['kurtosis'],
                    ['red'],
                    ['o'],
                    title,
                    'Feature Index',
                    'Kurtosis',
                    filename)
        
        '''

        all_mses, rng = self.reconstruction_error(X_train_scl, IncrementalPCA)
        
        title = "Reconstruction Error (ICA)"
        name = 'ica_rec_err'
        filename = './' + self.out_dir + '/' + name + '.png'
        
        plot_series(rng,
                    [all_mses.mean(0)],
                    [all_mses.std(0)],
                    ['mean square error'],
                    ['red'],
                    ['o'],
                    title,
                    'Number of Features',
                    'Mean Square Error',
                    filename)
        
def main():
    print('Running part 2')
    p = part2()
    
    t0 = timer()
    
    p.ica_wine()
    p.pca_wine()
    
    print("done in %0.3fs seconds" % (timer() - t0))

if __name__== '__main__':
    main()
    
