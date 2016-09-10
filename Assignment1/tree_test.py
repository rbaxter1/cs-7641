import io
import pydotplus

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import StratifiedKFold

from plot_curves import *

class rb_tree_test:
    
    def __init__(self, x_train, x_test, y_train, y_test, x_col_names, data_label, cv):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.x_col_names = x_col_names
        self.data_label = data_label
        self.cv = cv

    def run_cv_model(self, max_depth=3, criterion='entropy', do_plot=True):
        
        # use k-fold cross validation
        
        # Supported criteria are gini for the Gini impurity and entropy for the information gain.
        tree = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=0)
        
        # resample the test data without replacement. This means that each data point is part of a test a
        # training set only once. (paraphrased from Raschka p.176). In Stratified KFold, the features are
        # evenly disributed such that each test and training set is an accurate representation of the whole
        # this is the 0.17 version
        #kfold = StratifiedKFold(y=self.y_train, n_folds=self.cv, random_state=0)
        
        # this is the 0.18dev version
        skf = StratifiedKFold(n_splits=self.cv, random_state=0)
        
        # do the cross validation
        train_scores = []
        test_scores = []
        #for k, (train, test) in enumerate(kfold):
        for k, (train, test) in enumerate(skf.split(X=self.x_train, y=self.y_train)):
            
            # run the learning algorithm
            tree.fit(self.x_train[train], self.y_train[train])
            train_score = tree.score(self.x_train[test], self.y_train[test])
            train_scores.append(train_score)
            test_score = tree.score(self.x_test, self.y_test)
            test_scores.append(test_score)
            print('Fold:', k+1, ', Training score:', train_score, ', Test score:', test_score)
        
        train_score = np.mean(train_scores)
        print('Training score is', train_score)
        
        test_score = np.mean(test_scores)
        print('Test score is', test_score)
        
        if do_plot:
            self.__plot_learning_curve(tree)
            
        return train_score, test_score  
            
    def run_model(self, params, max_depth=3, criterion='entropy', do_plot=True):
        
        # Supported criteria for tree are gini for the Gini impurity and entropy for the information gain.
        tree = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=0)
        tree.fit(self.x_train, self.y_train)
        
        # export a graphical representation of the tree
        dot_data = io.StringIO()
        export_graphviz(tree,
                        out_file=dot_data,
                        feature_names=self.x_col_names)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        fn = './output/' + self.data_label + '_graph.pdf'
        graph.write_pdf(fn)
        
        # check model accuracy
        '''
        y_train_pred = tree.predict(self.x_train)
        train_acc = accuracy_score(self.y_train, y_train_pred)
        print('Training accuracy score is', train_acc)
        
        y_test_pred = tree.predict(self.x_test)
        test_acc = accuracy_score(self.y_test, y_test_pred)
        print('Test accuracy score is', test_acc)
        '''
        # no difference from above
        train_score = tree.score(self.x_train, self.y_train)
        print('Training score is', train_score)
        
        test_score = tree.score(self.x_test, self.y_test)
        print('Test score is', test_score)
        
        if do_plot:
            self.__plot_learning_curve(tree)
            self.__plot_decision_boundaries(tree)
        
        return train_score, test_score
        
    def __plot_learning_curve(self, estimator):
        plc = rb_plot_curves()
        plc.plot_learning_curve(estimator, self.x_train, self.y_train, self.cv, self.data_label)

    def plot_validation_curve(self, max_depth=3, criterion='entropy'):
        #for criterion in ['entropy', 'gini']:
        for criterion in ['gini']:        
            estimator = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
            param_names = ['max_depth']
            param_ranges = [np.arange(1,30,1)]
            data_label = criterion + '_' + self.data_label
            plc = rb_plot_curves()
            for i in range(len(param_names)):
                
                param_name = param_names[i]
                param_range = param_ranges[i]
                
                plc.plot_validation_curve(estimator, self.x_train, self.y_train,
                                          self.cv, data_label, 
                                          param_range, param_name)
     
    def plot_validation_curve2(self, max_depth=3, criterion='entropy'):
        #for criterion in ['entropy', 'gini']:
        for criterion in ['gini']:
            
            #params_dict = {"min_impurity_split": [0.2, 0.5, 0.8, 1.1, 1.4,
            #                          1.7, 2.0, 2.3, 2.6, 2.9, 3.2]}
            
            estimator = DecisionTreeClassifier(criterion=criterion)
            param_names = ['max_depth',
                           'min_impurity_split']
            
            param_ranges = [np.arange(1,30,1),
                            np.arange(0, 0.25, 0.005)]
            
            data_label = criterion + '_' + self.data_label
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