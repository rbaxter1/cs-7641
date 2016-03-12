import io
import pydotplus

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.cross_validation import StratifiedKFold, cross_val_score, cross_val_predict, train_test_split
#from sklearn.metrics import accuracy_score


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
        
        # resample the test data without replacement. This means that each data point is part of a test a
        # training set only once. (paraphrased from Raschka p.176). In Stratified KFold, the features are
        # evenly disributed such that each test and training set is an accurate representation of the whole
        kfold = StratifiedKFold(y=self.y_train, n_folds=self.cv, random_state=0)
        
        # Supported criteria are gini for the Gini impurity and entropy for the information gain.
        tree = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=0)
        
        # do the cross validation
        train_scores = []
        test_scores = []
        for k, (train, test) in enumerate(kfold):
            
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
            
    def run_model(self, max_depth=3, criterion='entropy', do_plot=True):
        
        # Supported criteria for tree are gini for the Gini impurity and entropy for the information gain.
        tree = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=0)
        tree.fit(self.x_train, self.y_train)
        
        # export a graphical representation of the tree
        dot_data = io.StringIO()
        export_graphviz(tree,
                        out_file=dot_data,
                        feature_names=self.x_col_names)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        fn = self.data_label + '_graph.pdf'
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
        
        return train_score, test_score
        
    def __plot_learning_curve(self, estimator):
        plc = rb_plot_curves()
        plc.plot_learning_curve(estimator, self.x_train, self.y_train, self.cv, self.data_label)

    def plot_validation_curve(self):
        for criterion in ['entropy', 'gini']:
                
            estimator = DecisionTreeClassifier(criterion=criterion)
            param_names = ['max_depth']
            param_ranges = [range(1,50)]
            data_label = criterion + '_' + self.data_label
            plc = rb_plot_curves()
            for i in range(len(param_names)):
                
                param_name = param_names[i]
                param_range = param_ranges[i]
                
                plc.plot_validation_curve(estimator, self.x_train, self.y_train,
                                          self.cv, data_label, 
                                          param_range, param_name)