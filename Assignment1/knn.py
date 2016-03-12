import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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
import itertools as iter



class rb_knn:
    
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
        
        knn = KNeighborsClassifier()
        knn.fit(x_train_std, y_train)
        
        
        
        # (Raschka, p.52)
        y_pred = knn.predict(x_test)
        ms = (y_test != y_pred).sum()
        print('Misclassified samples:', ms)
    
        # sklearn has a bunch of performance metrics, for example
        acc = accuracy_score(y_test, y_pred)
        print('Accuracy is', acc)        

        #x_combined = np.vstack((x_train_std, x_test_std))
        #y_combined = np.hstack((y_train, y_test))
        #    plot_decision_regions(X=x_combined, y=y_combined, clf=tree) #test_idx=range(105,150)
        #plot_decision_regions(x_combined, y_combined, clf=knn)
            #plt.xlabel('xxx')
            #plt.ylabel('quality')
        #    plt.show()        
        
                
        
        #Z = pipe_knn.predict(np.c_[xx.ravel(), yy.ravel()])
        
        # we need to standardize the data for the KNN learner
        pipe_knn = Pipeline([ ('scl', StandardScaler() ),
                              ('clf', KNeighborsClassifier())])
    
        # test it: this should match the non-pipelined call
        pipe_knn.fit(x_train, y_train)
        print('Test accuracy:', pipe_knn.score(x_test, y_test))        
        
        
        plt.clf()
        train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_knn,
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
        plt.title("Learning Curve with KNN")
        plt.xlabel('Number of training samples')
        plt.ylabel('Accurancy')
        plt.legend(loc='lower right')
        fn = self.data_label + '_learncurve.png'
        plt.savefig(fn)

        
        # plot the validation curves
        plt.clf()
        param_range = range(1,50)
        
        train_scores, test_scores = validation_curve(estimator=pipe_knn,
                                                     X=x_train,
                                                     y=y_train, 
                                                     param_name='clf__n_neighbors', 
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
        plt.title("Validation Curve with KNN")
        plt.xlabel('Max Depth')
        plt.ylabel('Accurancy')
        plt.legend(loc='lower right')
        fn = self.data_label + '_validationcurve.png'
        plt.savefig(fn)        

    
        # source: http://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html
        # these decision boundaries charts are great, but how to visualize higher dimensions?
        '''
        X = x_train_std
        y = y_train
        
        n_neighbors = 5
                
        h = .02  # step size in the mesh
        
        # Create color maps
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
        
        for weights in ['uniform', 'distance']:
            # we create an instance of Neighbours Classifier and fit the data.
            clf = KNeighborsClassifier(n_neighbors, weights=weights)
            clf.fit(X, y)
        
            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, m_max]x[y_min, y_max].
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        
            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            plt.figure()
            plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
        
            # Plot also the training points
            plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())
            plt.title("3-Class classification (k = %i, weights = '%s')"
                      % (n_neighbors, weights))
        
        plt.show()
        '''