from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import Imputer, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from timeit import default_timer as timer
from data_helper import *

class validation_curves:
    def __init__(self):
        pass
    
    def gridSearch2(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test =  dh.load_wine_data()

        pipeline = Pipeline([ ('scl', StandardScaler() ),
                              ('clf', KNeighborsClassifier(n_neighbors=n_neighbors,
                                                           leaf_size=leaf_size,
                                                           p=p, n_jobs=-1))])
        
        parameters = {'clf__n_neighbors': {'param_value': np.arange(1, 500, 10), 'reverse_xaxis': False},
                      'clf__leaf_size': {'param_value': np.arange(0, 50, 1), 'reverse_xaxis': False},
                      'clf__p': {'param_value': np.arange(0, 50, 1), 'reverse_xaxis': False}
                      }
        
        grid_search = GridSearchCV(pipeline, parameters, n_jobs=1, verbose=1)
        
        print("Performing grid search...")
        print("pipeline:", [name for name, _ in pipeline.steps])
        print("parameters:")
        print(parameters)
        t0 = timer()
        grid_search.fit(X_train, y_train)
        print("done in %0.3fs" % (timer() - t0))
        print()
    
        print("Best score: %0.3f" % grid_search.best_score_)
        print("Best parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))
            
    def run(self, X_train, X_test, y_train, y_test, dataset):
        
        params_dict ={'clf__n_neighbors': {'param_value': np.arange(1, 500, 10), 'reverse_xaxis': True},
                      'clf__leaf_size': {'param_value': np.arange(1, 20, 1), 'reverse_xaxis': False},
                      'clf__p': {'param_value': np.arange(1, 20, 1), 'reverse_xaxis': False}
                      }
        
        learner_name = 'KNN'
        
        for param_name in params_dict.keys():
            print(param_name)
                
            x = []
            in_sample_avg_errors = []
            std_in_sample_errors = []
            out_of_sample_avg_errors = []
            std_out_of_sample_errors = []
            avg_num_nodes = []
            std_num_nodes = []
            
            for param_value in params_dict[param_name]['param_value']:
                print(param_value)
                
                clf = Pipeline([ ('scl', StandardScaler() ),
                              ('clf', KNeighborsClassifier(n_jobs=-1))])
                params = clf.get_params()
                params[param_name] = param_value
                clf.set_params(**params)
                
                # this is the 0.18dev version
                skf = StratifiedKFold(n_splits=5, random_state=0)
                
                out_sample_errors = []
                in_sample_errors = []
                num_nodes = []
                
                # do the cross validation
                for k, (train, test) in enumerate(skf.split(X=X_train, y=y_train)):
                    
                    # run the learning algorithm
                    clf.fit(X_train[train], y_train[train])
                    
                    # complexity
                    nnodes = 1 # clf.named_steps['clf'].radius
                    num_nodes.append(nnodes)
                    
                    # in sample
                    predicted_values = clf.predict(X_train[train])
                    in_sample_mse = mean_squared_error(y_train[train], predicted_values)
                    in_sample_errors.append(in_sample_mse)
                    
                    # out of sample
                    predicted_values = clf.predict(X_train[test])
                    out_sample_mse = mean_squared_error(y_train[test], predicted_values)
                    out_sample_errors.append(out_sample_mse)

                    print('Fold:', k+1, ', Validation error: ', out_sample_mse, ', Training error: ', in_sample_mse, ', Tree nodes: ', nnodes)
                   
                # out of sample (test)
                avg_test_err = np.mean(out_sample_mse)
                test_err_std = np.std(out_sample_mse)
                out_of_sample_avg_errors.append(avg_test_err)
                std_out_of_sample_errors.append(test_err_std)
                
                # in sample (train)
                avg_train_err = np.mean(in_sample_mse)
                train_err_std = np.std(in_sample_mse)
                in_sample_avg_errors.append(avg_train_err)
                std_in_sample_errors.append(train_err_std)
                
                # complexity (num nodes)
                avg_nodes = np.mean(num_nodes)
                std_nodes = np.std(num_nodes)
                avg_num_nodes.append(avg_nodes)
                std_num_nodes.append(std_nodes)
                
                print('Avg validation MSE: ', avg_test_err, '+/-', test_err_std,
                      'Avg training MSE: ', avg_train_err, '+/-', train_err_std,
                      'Avg num tree nodes: ', avg_nodes, '+/-', std_nodes)
                
                x.append(param_value)
                
            # prepare
            param_range = x
            train_mean = np.array(in_sample_avg_errors)
            train_std = np.array(std_in_sample_errors)
            test_mean = np.array(out_of_sample_avg_errors)
            test_std = np.array(std_out_of_sample_errors)
            nodes_mean = np.array(avg_num_nodes)
            nodes_std = np.array(std_num_nodes)
            save_path= './output/'
            rev_axis = params_dict[param_name]['reverse_xaxis']

            # plot
            plt.cla()    
            plt.clf()
            
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()

            l1 = ax1.plot(param_range, train_mean,
                          color='blue', marker='o',
                          markersize=5,
                          label='training error')
            
            ax1.fill_between(param_range,
                             train_mean + train_std,
                             train_mean - train_std,
                             alpha=0.15, color='blue')
            
            l2 = ax1.plot(param_range, test_mean,
                          color='green', marker='s',
                          markersize=5, linestyle='--',
                          label='validation error')
            
            ax1.fill_between(param_range,
                             test_mean + test_std,
                             test_mean - test_std,
                             alpha=0.15, color='green')
            
            l3 = ax2.plot(param_range, nodes_mean,
                          color='red', marker='o',
                          markersize=5,
                          label='node count')
            
            ax2.fill_between(param_range,
                             nodes_mean + nodes_std,
                             nodes_mean - nodes_std,
                             alpha=0.15, color='red')
            
            ax1.set_xlabel(param_name)
            ax1.set_ylabel('Mean Squared Error')
            ax2.set_ylabel('Node Count')

            if (rev_axis):
                ax1.invert_xaxis()
                
            plt.grid()
            plt.title("%s: Training, Validation Error (left)\nand Node Count (right) Versus %s" % (learner_name, param_name))
            
            lns = l1+l2+l3
            labs = [l.get_label() for l in lns]
            ax1.legend(lns, labs, loc='center right')

            fn = save_path + learner_name + '_' + param_name + '_validation.png'
            plt.savefig(fn)
            
if __name__ == "__main__":
    vc = validation_curves()
    #vc.gridSearch2()
    
    dh = data_helper()
    
    X_train, X_test, y_train, y_test =  dh.load_titanic_data()
    vc.run(X_train, X_test, y_train, y_test, 'titanic')          
        
    X_train, X_test, y_train, y_test =  dh.load_wine_data()
    vc.run(X_train, X_test, y_train, y_test, 'wine')
              
        
        
        
        
        
