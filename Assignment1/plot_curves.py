import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import itertools as it
import numpy as np

from sklearn.model_selection import learning_curve, validation_curve

class rb_plot_curves:
    
    def __init__(self, save_path='./output/'):
        self.save_path = save_path

    def plot_learning_curve(self, estimator, x_train, y_train, cv, data_label, n_jobs=-1):
            
        # plot the learning curves using sklearn and matplotlib
        plt.clf()
        train_sizes, train_scores, test_scores = learning_curve(estimator=estimator,
                                                                X=x_train,
                                                                y=y_train,
                                                                cv=cv,
                                                                n_jobs=n_jobs)
        
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
        plt.title('Learning Curve')
        plt.xlabel('Number of training samples')
        plt.ylabel('Accurancy')
        plt.legend(loc='lower right')
        fn = self.save_path + data_label + '_learncurve.png'
        plt.savefig(fn)
    
    
    def plot_validation_curve(self, estimator, x_train, y_train, cv, data_label, param_range, param_name, n_jobs=-1):
        
        # plot the validation curves
        plt.clf()
        
        train_scores, test_scores = validation_curve(estimator=estimator,
                                                     X=x_train,
                                                     y=y_train, 
                                                     param_name=param_name, 
                                                     param_range=param_range, 
                                                     cv=cv,
                                                     n_jobs=n_jobs)
    
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
        plt.title("Validation Curve")
        plt.xlabel(param_name)
        plt.ylabel('Accurancy')
        plt.legend(loc='lower right')
        fn = self.save_path + data_label + '_' + param_name + '_validationcurve.png'
        plt.savefig(fn)
        
        
    def plot_decision_boundaries(self, estimator, features, y, data_label):
        
        
        # pinched from http://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html
        # run this for each pairwise feature
        feature_pairs = it.combinations(features.columns, 2)
        for k, p in enumerate(feature_pairs):
            
            plt.clf()

            X = features[list(p)].values
            
            h = .02  # step size in the mesh
            
            # Create color maps
            cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
            cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
            
            estimator.fit(X, y)
        
            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, m_max]x[y_min, y_max].
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
            Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])
        
            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            plt.figure()
            plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
        
            # Plot also the training points
            plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())
            plt.title("Decision boundaries (%s, %s)" % (p[0], p[1]))
            
            fn = self.save_path + data_label + '_' + str(k) + '_decision_boundary.png'
            plt.savefig(fn)

        