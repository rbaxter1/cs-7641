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

'''
2. Apply the dimensionality reduction algorithms to the two datasets and describe what you see.
'''        
class part2():
    def __init__(self):
        self.out_dir = 'output_part2'

    def pca_wine(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_wine_data()
        self.pca_analysis(X_train, X_test, y_train, y_test, 'Wine')
    
    def ica_wine(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_wine_data()
        self.ica_analysis(X_train, X_test, y_train, y_test, 'Wine')
    
    def rp_wine(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_wine_data()
        self.rp_analysis(X_train, X_test, y_train, y_test, 'Wine')

    def lda_wine(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_wine_data()
        self.lda_analysis(X_train, X_test, y_train, y_test, 'Wine')
    
    def pca_nba(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_nba_data()
        self.pca_analysis(X_train, X_test, y_train, y_test, 'NBA')
        
    def ica_nba(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_nba_data()
        self.ica_analysis(X_train, X_test, y_train, y_test, 'NBA')

    def rp_nba(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_nba_data()
        self.rp_analysis(X_train, X_test, y_train, y_test, 'NBA')
        
    def lda_nba(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_nba_data()
        self.lda_analysis(X_train, X_test, y_train, y_test, 'NBA')

    def reconstruction_error(self, X_train_scl, cls):
        rng = range(1, X_train_scl.shape[1]+1)
        
        all_mses = np.ndarray([0, len(rng)])
        for n in range(100):
            mses = []
            for i in rng:
                dr = cls(n_components=i)
                X_transformed = dr.fit_transform(X_train_scl)
                
                X_projected = dr.inverse_transform(X_transformed)
                mse = mean_squared_error(X_train_scl, X_projected)
                mses.append(mse)
                print(i, mse)
            
            all_mses = np.vstack([all_mses, mses])
            
        return all_mses, rng
    
    def plot_scatter(self, X, y, title, filename, f0_name='feature 1', f1_name='feature 2', x0_i=0, x1_i=1):
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

    def plot_explained_variance(self, pca, title, filename):
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
    
    def lda_analysis(self, X_train, X_test, y_train, y_test, data_set_name):
        scl = RobustScaler()
        X_train_scl = scl.fit_transform(X_train)
        X_test_scl = scl.transform(X_test)
        
        ##
        ## Plots
        ##
        ph = plot_helper()
        
        scores = []
        rng = range(1, X_train_scl.shape[1]+1)
        for i in rng:
            lda = LinearDiscriminantAnalysis(n_components=i)
            cv = KFold(X_train_scl.shape[0], 3, shuffle=True)
            
            # cross validation
            cv_scores = []
            for (train, test) in cv:
                lda.fit(X_train_scl[train], y_train[train])
                score = lda.score(X_train_scl[test], y_train[test])
                cv_scores.append(score)
                
            mean_score = np.mean(cv_scores)
            scores.append(mean_score)
            print(i, mean_score)
            
        ##
        ## Score Plot
        ##
        title = 'Score Summary Plot (LDA) for ' + data_set_name
        name = data_set_name.lower() + '_lda_score'
        filename = './' + self.out_dir + '/' + name + '.png'
                    
        ph.plot_series(rng,
                       [scores],
                       [None],
                       ['cross validation score'],
                       cm.viridis(np.linspace(0, 1, 1)),
                       ['o'],
                       title,
                       'n_components',
                       'Score',
                       filename)
        
        
    def rp_analysis(self, X_train, X_test, y_train, y_test, data_set_name):
        scl = RobustScaler()
        X_train_scl = scl.fit_transform(X_train)
        
        ks = []
        for i in range(1000):
            ##
            ## Random Projection
            ##
            rp = GaussianRandomProjection(n_components=X_train_scl.shape[1])
            rp.fit(X_train_scl)
            X_train_rp = rp.transform(X_train_scl)
            
            ks.append(kurtosis(X_train_rp))
            
        mean_k = np.mean(ks, 0)
            
        ##
        ## Plots
        ##
        ph = plot_helper()
        
        title = 'Kurtosis (Randomized Projection) for ' + data_set_name
        name = data_set_name.lower() + '_rp_kurt'
        filename = './' + self.out_dir + '/' + name + '.png'
        
        ph.plot_simple_bar(np.arange(1, len(mean_k)+1, 1),
                           mean_k,
                           np.arange(1, len(mean_k)+1, 1).astype('str'),
                           'Feature Index',
                           'Kurtosis',
                           title,
                           filename)
        
        
    def pca_analysis(self, X_train, X_test, y_train, y_test, data_set_name):
        scl = RobustScaler()
        X_train_scl = scl.fit_transform(X_train)
        X_test_scl = scl.transform(X_test)
        
        ##
        ## PCA
        ##
        pca = PCA(n_components=X_train_scl.shape[1], svd_solver='full')
        X_pca = pca.fit_transform(X_train_scl)
        
        ##
        ## Plots
        ##
        ph = plot_helper()
        
        ##
        ## Explained Variance Plot
        ##
        title = 'Explained Variance (PCA) for ' + data_set_name
        name = data_set_name.lower() + '_pca_evar_err'
        filename = './' + self.out_dir + '/' + name + '.png'        
        self.plot_explained_variance(pca, title, filename)

        ##
        ## Reconstruction Error
        ##
        all_mses, rng = self.reconstruction_error(X_train_scl, PCA)
        
        title = 'Reconstruction Error (PCA) for ' + data_set_name
        name = data_set_name.lower() + '_pca_rec_err'
        filename = './' + self.out_dir + '/' + name + '.png'
        ph.plot_series(rng,
                    [all_mses.mean(0)],
                    [all_mses.std(0)],
                    ['mse'],
                    ['red'],
                    ['o'],
                    title,
                    'Number of Features',
                    'Mean Squared Error',
                    filename)
        
        
        ##
        ## Manually compute eigenvalues
        ## 
        cov_mat = np.cov(X_train_scl.T)
        eigen_values, eigen_vectors = np.linalg.eig(cov_mat)
        print(eigen_values)
        sorted_eigen_values = sorted(eigen_values, reverse=True)

        title = 'Eigen Values (PCA) for ' + data_set_name
        name = data_set_name.lower() + '_pca_eigen'
        filename = './' + self.out_dir + '/' + name + '.png'
        
        ph.plot_simple_bar(np.arange(1, len(sorted_eigen_values)+1, 1),
                           sorted_eigen_values,
                           np.arange(1, len(sorted_eigen_values)+1, 1).astype('str'),
                           'Principal Components',
                           'Eigenvalue',
                           title,
                           filename)
        
        ## TODO Factor this out to new method
        ##
        ## Scatter
        ##
        '''
        pca = PCA(n_components=2, svd_solver='full')
        X_pca = pca.fit_transform(X_train_scl)

        title = 'PCA Scatter: Wine'
        filename = './' + self.out_dir + '/wine_pca_sc.png'    
        self.plot_scatter(X_pca, y_train, title, filename, f0_name='feature 1', f1_name='feature 2', x0_i=0, x1_i=1)    
        '''
        
    def ica_analysis(self, X_train, X_test, y_train, y_test, data_set_name):
        scl = RobustScaler()
        X_train_scl = scl.fit_transform(X_train)
        X_test_scl = scl.transform(X_test)
        
        ##
        ## ICA
        ##
        ica = FastICA(n_components=X_train_scl.shape[1])
        X_ica = ica.fit_transform(X_train_scl)
        
        ##
        ## Plots
        ##
        ph = plot_helper()

        kurt = kurtosis(X_ica)
        print(kurt)
        
        title = 'Kurtosis (FastICA) for ' + data_set_name
        name = data_set_name.lower() + '_ica_kurt'
        filename = './' + self.out_dir + '/' + name + '.png'
        
        ph.plot_simple_bar(np.arange(1, len(kurt)+1, 1),
                           kurt,
                           np.arange(1, len(kurt)+1, 1).astype('str'),
                           'Feature Index',
                           'Kurtosis',
                           title,
                           filename)
        
        
def main():
    print('Running part 2')
    p = part2()
    
    t0 = time()
    
    p.pca_wine()
    p.pca_nba()
    
    p.ica_wine()
    p.ica_nba()
    
    p.rp_wine()
    p.rp_nba()
    
    p.lda_nba()
    p.lda_wine()
    
    print("done in %0.3f seconds" % (time() - t0))

if __name__== '__main__':
    main()
    
