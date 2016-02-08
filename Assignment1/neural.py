import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz, DecisionTreeRegressor
from sklearn.cross_validation import StratifiedKFold, cross_val_score, cross_val_predict, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.learning_curve import learning_curve, validation_curve
from sklearn.pipeline import Pipeline

from mlxtend.evaluate import plot_decision_regions, plot_learning_curves

import io
import pydotplus
import itertools as iter



class rb_neural:
    
    def __init__(self, x, y, cols, cv, data_label, max_depth=3, test_size=0.3):
        self.max_depth = max_depth
        self.test_size = test_size
        self.x = x
        self.y = y
        self.cols = cols
        self.data_label = data_label
        self.cv = cv

    
    def run(self):
        
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=self.test_size, random_state=0)
        
        stdsc = StandardScaler()
        x_train_std = stdsc.fit_transform(x_train)
        x_test_std = stdsc.fit_transform(x_test)        
        
        clf = MLPClassifier()
        clf.fit(x_train_std, y_train)
        
        
        
        # (Raschka, p.52)
        y_pred = clf.predict(x_test)
        ms = (y_test != y_pred).sum()
        print('Misclassified samples:', ms)
    
        # sklearn has a bunch of performance metrics, for example
        acc = accuracy_score(y_test, y_pred)
        print('Accuracy is', acc)        

        # we need to standardize the data for the KNN learner
        pipe_clf = Pipeline([ ('scl', StandardScaler() ),
                              ('clf', MLPClassifier())])
    
        # test it: this should match the non-pipelined call
        pipe_clf.fit(x_train, y_train)
        print('Test accuracy:', pipe_clf.score(x_test, y_test))        
        
        
        plt.clf()
        train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_clf,
                                                                X=x_train,
                                                                y=y_train,
                                                                cv=self.cv,
                                                                n_jobs=-1)
    
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
    
        plt.plot(train_sizes, train_mean,
                 color='blue', marker='o',
                 markersize=5,
                 label='training accuracy')
    
        plt.fill_between(train_sizes,
                         train_mean + train_std,
                         train_mean - train_std,
                         alpha=0.15, color='blue')
    
        plt.plot(train_sizes, test_mean,
                 color='green', marker='s',
                 markersize=5, linestyle='--',
                 label='validation accuracy')        
    
        plt.fill_between(train_sizes,
                         test_mean + test_std,
                         test_mean - test_std,
                         alpha=0.15, color='green')
    
        plt.grid()
        plt.title("Learning Curve with NueralNet")
        plt.xlabel('Number of training samples')
        plt.ylabel('Accurancy')
        plt.legend(loc='lower right')
        fn = self.data_label + '_learncurve.png'
        plt.savefig(fn)

        
        # plot the validation curves
        plt.clf()
        param_range = range(200,300)
        
        train_scores, test_scores = validation_curve(estimator=pipe_clf,
                                                     X=x_train,
                                                     y=y_train, 
                                                     param_name='clf__max_iter', 
                                                     param_range=param_range, 
                                                     cv=self.cv,
                                                     n_jobs=-1)
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        plt.plot(param_range, train_mean,
                 color='blue', marker='o',
                 markersize=5,
                 label='training accuracy')
        
        plt.fill_between(param_range,
                         train_mean + train_std,
                         train_mean - train_std,
                         alpha=0.15, color='blue')
        
        
        plt.plot(param_range, test_mean,
                 color='green', marker='s',
                 markersize=5, linestyle='--',
                 label='validation accuracy')
        
        plt.fill_between(param_range,
                         test_mean + test_std,
                         test_mean - test_std,
                         alpha=0.15, color='green')
        
        
        plt.grid()
        #plt.xscale('log')
        plt.title("Validation Curve with Neural Net")
        plt.xlabel('Max Depth')
        plt.ylabel('Accurancy')
        plt.legend(loc='lower right')
        fn = self.data_label + '_validationcurve.png'
        plt.savefig(fn)        

    