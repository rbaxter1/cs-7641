import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz, DecisionTreeRegressor
from sklearn.cross_validation import StratifiedKFold, cross_val_score, cross_val_predict, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.learning_curve import learning_curve, validation_curve
from sklearn.pipeline import Pipeline

#from mlxtend.evaluate import plot_decision_regions, plot_learning_curves

import io
import pydotplus



class rb_boost:
    
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
        
        #stdsc = StandardScaler()
        #x_train_std = stdsc.fit_transform(x_train)
        #x_test_std = stdsc.fit_transform(x_test)        
        
        tree = DecisionTreeClassifier(criterion='entropy', max_depth=1)
        
        clf = AdaBoostClassifier(base_estimator=tree, n_estimators=500, 
                                learning_rate=0.1,
                                random_state=0)
        
        tree.fit(x_train, y_train)
        y_train_pred = tree.predict(x_train)
        y_test_pred = tree.predict(x_test)
        
        acc = accuracy_score(y_train, y_train_pred)
        print('Accuracy on training is', acc)        

        acc = accuracy_score(y_test, y_test_pred)
        print('Accuracy on testing is', acc)        
        
        
        clf.fit(x_train, y_train)
        y_train_pred = clf.predict(x_train)
        y_test_pred = clf.predict(x_test)
        
        acc = accuracy_score(y_train, y_train_pred)
        print('Accuracy on training is', acc)        

        acc = accuracy_score(y_test, y_test_pred)
        print('Accuracy on testing is', acc)              
        
        
        
        
        # we need to standardize the data for the KNN learner
        #pipe_clf = Pipeline([ ('scl', StandardScaler() ),
        #                      ('clf', SVC())])
    
        # test it: this should match the non-pipelined call
        #pipe_clf.fit(x_train, y_train)
        #print('Test accuracy:', pipe_clf.score(x_test, y_test))        
        
        
        # plot the learning curves using sklearn and matplotlib
        plt.clf()
        train_sizes, train_scores, test_scores = learning_curve(estimator=clf,
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
        plt.title("Learning Curve with Decision Tree")
        plt.xlabel('Number of training samples')
        plt.ylabel('Accurancy')
        plt.legend(loc='lower right')
        fn = self.data_label + '_boost_learncurve.png'
        plt.savefig(fn)
    
    
    
    
        # plot the validation curves
        plt.clf()
        param_range = range(1,10)
    
        train_scores, test_scores = validation_curve(estimator=clf,
                                                     X=x_train,
                                                     y=y_train, 
                                                     param_name='base_estimator__max_depth', 
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
        plt.title("Validation Curve with Decision Tree")
        plt.xlabel('Max Depth')
        plt.ylabel('Accurancy')
        plt.legend(loc='lower right')
        fn = self.data_label + '_boost_validationcurve.png'
        plt.savefig(fn)

    
        '''
        plt.clf()
        x_min = x_train[:, 0].min()
        x_max = x_train[:, 0].max()
        y_min = x_train[:, 1].min()
        y_max = x_train[:, 1].max()
        
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
        
        f, axarr = plt.subplots(1, 2,
                                 sharex='col',
                                 sharey='row',
                                 figsize=(8, 3))
        
        for idx, clf, tt in zip([0,1],
                                [tree, clf],
                                ['Decision Tree', 'AdaBoost']):
            
            clf.fit(x_train, y_train)
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            axarr[idx].contourf(xx, yy, Z, alpha=0.3)
            axarr[idx].scatter(x_train[y_train==0, 0],
                               x_train[y_train==0, 1],
                               c='blue',
                               marker='^')
            
            axarr[idx].scatter(x_train[y_train==1, 0],
                               x_train[y_train==1, 1],
                               c='red',
                               marker='o')
            
            axarr[idx].set_title(tt)
            axarr[0].set_ylabel('Alcohol', fontsize=12)
            plt.text(10.2, -1.2,
                     s='Hue',
                     ha='center',
                     va='center',
                     fontsize=12)
            
            plt.show()
            
        fn = self.data_label + '_', tt, '_boost_compare.png'
        plt.savefig(fn)            
        '''