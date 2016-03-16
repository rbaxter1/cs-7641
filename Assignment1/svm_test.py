import io
import pydotplus

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, Imputer
#from sklearn.metrics import accuracy_score


from plot_curves import *

class rb_svm_test:
    
    def __init__(self, x_train, x_test, y_train, y_test, x_col_names, data_label, cv):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.x_col_names = x_col_names
        self.data_label = data_label
        self.cv = cv

    def run_cv_model(self, C=1.0, degree=3, cache_size=200, do_plot=True):
        
        # use k-fold cross validation
        
        # we need to standardize the data for the KNN learner
        pipe_clf = Pipeline([ ('scl', StandardScaler() ),
                              ('clf', SVC(C=C, degree=degree, cache_size=cache_size))])
    
        # resample the test data without replacement. This means that each data point is part of a test a
        # training set only once. (paraphrased from Raschka p.176). In Stratified KFold, the features are
        # evenly disributed such that each test and training set is an accurate representation of the whole
        # this is the 0.17 version
        #kfold = StratifiedKFold(y=self.y_train, n_folds=self.cv, random_state=0)
        
        # this is the 0.18dev version
        skf = StratifiedKFold(n_folds=self.cv, random_state=0)
        
        # do the cross validation
        train_scores = []
        test_scores = []
        #for k, (train, test) in enumerate(kfold):
        for k, (train, test) in enumerate(skf.split(X=self.x_train, y=self.y_train)):
            
            # run the learning algorithm
            pipe_clf.fit(self.x_train[train], self.y_train[train])
            train_score = pipe_clf.score(self.x_train[test], self.y_train[test])
            train_scores.append(train_score)
            test_score = pipe_clf.score(self.x_test, self.y_test)
            test_scores.append(test_score)
            print('Fold:', k+1, ', Training score:', train_score, ', Test score:', test_score)
        
        train_score = np.mean(train_scores)
        print('Training score is', train_score)
        
        test_score = np.mean(test_scores)
        print('Test score is', test_score)
        
        if do_plot:
            self.__plot_learning_curve(pipe_clf)
            
        return train_score, test_score  
            
    def run_model(self, C=1.0, degree=3, cache_size=200, do_plot=True):
        
        # we need to standardize the data for the learner
        pipe_clf = Pipeline([ ('scl', StandardScaler() ),
                              ('clf', SVC(C=C, degree=degree, cache_size=cache_size))])
    
        # test it: this should match the non-pipelined call
        pipe_clf.fit(self.x_train, self.y_train)
        
        # check model accuracy
        train_score = pipe_clf.score(self.x_train, self.y_train)
        print('Training score is', train_score)
        
        test_score = pipe_clf.score(self.x_test, self.y_test)
        print('Test score is', test_score)
        
        if do_plot:
            self.__plot_learning_curve(pipe_clf)
            self.__plot_decision_boundaries(pipe_clf)
        
        return train_score, test_score
        
    def __plot_learning_curve(self, estimator):
        plc = rb_plot_curves()
        plc.plot_learning_curve(estimator, self.x_train, self.y_train, self.cv, self.data_label)

    def plot_validation_curve(self, C=1.0, degree=3, cache_size=200):

        estimator = Pipeline([ ('scl', StandardScaler() ),
                              ('clf', SVC(C=C, degree=degree, cache_size=cache_size))])
        param_names = ['clf__C']
        param_ranges = [np.arange(1.0,10.0,1.)]
        data_label = self.data_label
        plc = rb_plot_curves()
        for i in range(len(param_names)):
            
            param_name = param_names[i]
            param_range = param_ranges[i]
            
            plc.plot_validation_curve(estimator, self.x_train, self.y_train,
                                      self.cv, data_label, 
                                      param_range, param_name)
            
    def __plot_decision_boundaries(self, estimator):
        plc = rb_plot_curves()
        features = pd.DataFrame(self.x_train)
        features.columns = self.x_col_names
    
        plc.plot_decision_boundaries(estimator, features, self.y_train, self.data_label)