import matplotlib.pyplot as plt
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