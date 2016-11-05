import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn.model_selection import learning_curve, ShuffleSplit, StratifiedShuffleSplit, validation_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from data_helper import *

class plot_helper():
    def __init__():
        pass
    
    def extended_line_from_first_two_points(self, series):
        r = series[1]-series[0]
        lin = np.ones_like(series) * series[0]
        for i in range(1, lin.shape[0]):
            new = lin[i] + i * r
            if new <= 0:
                lin[i] = None
            else:
                lin[i] = new
                
        return lin

    def plot_series(self, x, y, y_std, y_lab, colors, markers, title, xlab, ylab, filename):
        
        plt.clf()
        plt.cla()
        fig, ax = plt.subplots()
        
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        
        for i in range(len(y)):
            plt.plot(x, y[i],
                     color=colors[i], marker=markers[i],
                     markersize=5,
                     label=y_lab[i])
            
            if None != y_std[i]:
                plt.fill_between(x,
                                 y[i] + y_std[i],
                                 y[i] - y_std[i],
                                 alpha=0.15, color=colors[i])
        
        plt.grid()
        plt.title(title)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.legend(loc='best')
        
        plt.savefig(filename)
        
        plt.close('all')
        
    
        
    # source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    def plot_learning_curve(self, estimator, title, X, y, ylim=None, cv=None,
                            n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5),
                            filename=None):
        
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()
    
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    
        plt.legend(loc="best")
        
        if filename != None:
            plt.savefig(filename)
            
        return plt

    
    def plot_validation_curve(self, X_train, X_test, y_train, y_test, pipeline, param_name, param_range, title, filename, cv=4, param_range_plot=None):
        train_scores, test_scores = validation_curve(estimator=pipeline,
                                                     X=X_train,
                                                     y=y_train, 
                                                     param_name=param_name, 
                                                     param_range=param_range,
                                                     cv=cv)
        
        #train_scores = 1. - train_scores
        #test_scores = 1. - test_scores
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        if param_range_plot != None:
            param_range = param_range_plot
        
        plot_series(param_range,
                    [train_mean, test_mean],
                    [train_std, test_std],
                    ['training accuracy', 'validation accuracy'],
                    ['blue', 'green'],
                    ['o', 's'],
                    title,
                    param_name,
                    'Accuracy',
                    filename)

if __name__ == '__main__':
    print('these are not the droid you are looking for')