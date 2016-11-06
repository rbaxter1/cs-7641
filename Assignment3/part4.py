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
4. Apply the dimensionality reduction algorithms to one of your datasets from assignment #1
(if you've reused the datasets from assignment #1 to do experiments 1-3 above then you've
already done this) and rerun your neural network learner on the newly projected data.
'''        
class part4():
    def __init__(self):
        self.out_dir = 'output_part4'
        self.time_filename = './' + self.out_dir + '/time.txt'
        self.nn_time_filename = './' + self.out_dir + '/nn_time.txt'

    def run(self):
        print('Running part 4')
    
        with open(self.time_filename, 'w') as text_file:
            
            t0 = time()
            self.nn_pca_wine()
            text_file.write('nn_pca_wine: %0.3f seconds\n' % (time() - t0))
            
            t0 = time()
            self.nn_ica_wine()
            text_file.write('nn_ica_wine: %0.3f seconds\n' % (time() - t0))
            
            t0 = time()
            self.nn_rp_wine()
            text_file.write('nn_rp_wine: %0.3f seconds\n' % (time() - t0))
            
            t0 = time()
            self.nn_lda_wine()
            text_file.write('nn_lda_wine: %0.3f seconds\n' % (time() - t0))
            
        
    def nn_pca_wine(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_wine_data_pca_best()
        self.nn_analysis(X_train, X_test, y_train, y_test, 'Wine', 'Neural Network PDA')
        
    def nn_ica_wine(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_wine_data_ica_best()
        self.nn_analysis(X_train, X_test, y_train, y_test, 'Wine', 'Neural Network IDA')
        
    def nn_rp_wine(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_wine_data_rp_best()
        self.nn_analysis(X_train, X_test, y_train, y_test, 'Wine', 'Neural Network RP')
        
    def nn_lda_wine(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_wine_data_lda_best()
        self.nn_analysis(X_train, X_test, y_train, y_test, 'Wine', 'Neural Network LDA')
    
    def nn_analysis(self, X_train, X_test, y_train, y_test, data_set_name, analysis_name='Neural Network'):
        
        clf = MLPClassifier(activation='relu',
                            learning_rate='constant',
                            shuffle=True,
                            solver='adam',
                            random_state=0,
                            max_iter=1000,
                            batch_size=60)
    
        with open(self.nn_time_filename, 'w') as text_file:
            t0 = time()
            clf.fit(X_train, y_train)
            text_file.write(analysis_name.lower() + ' fit time: %0.3f seconds\n' % (time() - t0))
            
            t0 = time()
            y_pred = clf.predict(X_test)
            text_file.write(analysis_name.lower() + ' predict time: %0.3f seconds\n' % (time() - t0))
            
        cv = StratifiedShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
        
        title = 'Learning Curve (' + analysis_name + ') for ' + data_set_name
        name = data_set_name.lower() + '_' + analysis_name.lower() + '_nn.png'
        filename = './' + self.out_dir + '/' + name
        
        ##
        ## Plots
        ##
        ph = plot_helper()
        
        ##
        ## Learning Curve
        ##
        ph.plot_learning_curve(clf, title, X_train, y_train, ylim=None, cv=cv, n_jobs=-1, filename=filename)
        
        
def main():    
    p = part4()
    p.run()

if __name__== '__main__':
    main()
    
