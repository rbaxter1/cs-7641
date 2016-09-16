from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import Imputer, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_graphviz

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA

import numpy as np
import pandas as pd
from data_helper import *

from timeit import default_timer as timer

class validation_curves:
    def __init__(self):
        pass
    
    def gridSearch2(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test =  dh.load_wine_data()

        pipeline = Pipeline([#('scl', StandardScaler()),
                             ('clf', DecisionTreeClassifier(random_state=0))])
        
        parameters = {'clf__criterion': ('gini', 'entropy'),
                      'clf__min_impurity_split': np.arange(0, 0.5, 0.01),
                      'clf__max_depth': np.arange(1, 40, 1)#,
                      #'clf__min_samples_split': np.arange(1, 200, 5),
                      #'clf__min_samples_leaf': np.arange(2, 200, 5),
                      #'clf__min_weight_fraction_leaf': np.arange(0.0, 0.5, 0.05),
                      #'clf__max_leaf_nodes': np.arange(2, 300, 5)
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
        
        params_dict = {
            'min_impurity_split': {'param_value': np.arange(0.0, 1.0, 0.02), 'reverse_xaxis': True},
            'max_depth': {'param_value': np.arange(1, 25, 1), 'reverse_xaxis': False},
            'min_samples_split': {'param_value': np.arange(1, 200, 5), 'reverse_xaxis': True},
            'min_samples_leaf': {'param_value': np.arange(2, 200, 5), 'reverse_xaxis': True},
            'max_leaf_nodes': {'param_value': np.arange(2, 225, 5), 'reverse_xaxis': False},
            'max_features': {'param_value': [1, 2, 3, 4, 5], 'reverse_xaxis': False}
            }
        
        ##'min_weight_fraction_leaf': np.arange(0.0, 0.5, 0.01),
        
        learner_name = 'Tree'
        
        for param_name in params_dict.keys():
            print(param_name)
                
            x = []
            in_sample_avg_errors = []
            std_in_sample_errors = []
            out_of_sample_avg_errors = []
            std_out_of_sample_errors = []
            avg_num_nodes = []
            std_num_nodes = []
            avg_fit_times = []
            std_fit_times = []
            avg_predict_times = []
            std_predict_times = []
            
            
            for param_value in params_dict[param_name]['param_value']:
                print(param_value)
                
                clf = DecisionTreeClassifier(criterion='entropy', random_state=0)
                params = clf.get_params()
                params[param_name] = param_value
                clf.set_params(**params)
                
                # this is the 0.18dev version
                skf = StratifiedKFold(n_splits=5, random_state=0)
                
                out_sample_errors = []
                in_sample_errors = []
                num_nodes = []
                fit_times = []
                predict_times = []
                
                # do the cross validation
                for k, (train, test) in enumerate(skf.split(X=X_train, y=y_train)):
                    
                    start = timer()
                    # run the learning algorithm
                    clf.fit(X_train[train], y_train[train])
                    end = timer()
                    tot_fit_time = (end - start) * 1000.
                    fit_times.append(tot_fit_time)
                    
                    # complexity
                    nnodes = clf.tree_.node_count
                    num_nodes.append(nnodes)
                    
                    # in sample
                    start = timer()
                    predicted_values = clf.predict(X_train[train])
                    end = timer()
                    tot_in_sample_predict_time = (end - start) * 1000.
                    predict_times.append(tot_in_sample_predict_time)
                    
                    in_sample_mse = mean_squared_error(y_train[train], predicted_values)
                    in_sample_errors.append(in_sample_mse)
                    
                    # out of sample
                    start = timer()
                    predicted_values = clf.predict(X_train[test])
                    end = timer()
                    tot_out_sample_predict_time = (end - start) * 1000.
                    predict_times.append(tot_out_sample_predict_time)
                    out_sample_mse = mean_squared_error(y_train[test], predicted_values)
                    out_sample_errors.append(out_sample_mse)

                    print('Fold:', k+1, ', Validation error: ', out_sample_mse,
                          ', Training error: ', in_sample_mse, ', Tree nodes: ', nnodes,
                          ', fit time: ', tot_fit_time, ', in sample predict time: ', tot_in_sample_predict_time,
                          ', out of sample predict time: ', tot_out_sample_predict_time)
                   
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
                
                # timings
                avg_fit_time = np.mean(fit_times)
                std_fit_time = np.std(fit_times)
                avg_fit_times.append(avg_fit_time)
                std_fit_times.append(std_fit_time)
                
                avg_predict_time = np.mean(predict_times)
                std_predict_time = np.std(predict_times)
                avg_predict_times.append(avg_predict_time)
                std_predict_times.append(std_predict_time)
                
                # complexity (num nodes) - how can this be universally applied?
                avg_nodes = np.mean(num_nodes)
                std_nodes = np.std(num_nodes)
                avg_num_nodes.append(avg_nodes)
                std_num_nodes.append(std_nodes)
                
                print('Avg validation MSE: ', avg_test_err, '+/-', test_err_std,
                      'Avg training MSE: ', avg_train_err, '+/-', train_err_std,
                      'Avg num tree nodes: ', avg_nodes, '+/-', std_nodes,
                      'Avg fit time: ', avg_fit_time, '+/-', std_fit_time,
                      'Avg predict time: ', avg_predict_time, '+/-', std_predict_time)
                
                x.append(param_value)
                
            # prepare
            param_range = x
            train_mean = np.array(in_sample_avg_errors)
            train_std = np.array(std_in_sample_errors)
            test_mean = np.array(out_of_sample_avg_errors)
            test_std = np.array(std_out_of_sample_errors)
            nodes_mean = np.array(avg_num_nodes)
            nodes_std = np.array(std_num_nodes)
            fit_time_mean = np.array(avg_fit_times)
            fit_time_std = np.array(std_fit_times)
            predict_time_mean = np.array(avg_predict_times)
            predict_time_std = np.array(std_predict_times)
            save_path= './output/'
            rev_axis = params_dict[param_name]['reverse_xaxis']

            # plot
            plt.cla()    
            plt.clf()
            
            if False:
                    
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
                
                '''
                l3 = ax2.plot(param_range, nodes_mean,
                              color='red', marker='o',
                              markersize=5,
                              label='node count')
                
                ax2.fill_between(param_range,
                                 nodes_mean + nodes_std,
                                 nodes_mean - nodes_std,
                                 alpha=0.15, color='red')
                '''
                
                l3 = ax2.plot(param_range, fit_time_mean,
                              color='gray', marker='3',
                              markersize=5,
                              label='fit time')
                
                ax2.fill_between(param_range,
                                 fit_time_mean + fit_time_std,
                                 fit_time_mean - fit_time_std,
                                 alpha=0.15, color='gray')
                
                l4 = ax2.plot(param_range, predict_time_mean,
                              color='orange', marker='4',
                              markersize=5,
                              label='predict time')
                
                ax2.fill_between(param_range,
                                 predict_time_mean + predict_time_std,
                                 predict_time_mean - predict_time_std,
                                 alpha=0.15, color='orange')
                
                ax1.set_xlabel(param_name)
                ax1.set_ylabel('Mean Squared Error')
                #ax2.set_ylabel('Node Count')
                ax2.set_ylabel('Time')
    
                plt.grid()
                #plt.title("%s: Training, Validation Error (left)\nand Node Count (right) Versus %s" % (learner_name, param_name))
                plt.title("%s: Training, Validation Error (left)\nand Timings (right) Versus %s" % (learner_name, param_name))
                
                lns = l1+l2+l3+l4
                labs = [l.get_label() for l in lns]
                ax1.legend(lns, labs, loc='center right')
                
                if (rev_axis):
                    ax1.invert_xaxis()
                    #ax2.invert_xaxis()
                    
                
                fn = save_path + dataset + '_' + learner_name + '_' + param_name + '_validation.png'
                plt.savefig(fn)
            
            
            if True:
                host = host_subplot(111, axes_class=AA.Axes)
                plt.subplots_adjust(right=0.75)
            
                par1 = host.twinx()
                par2 = host.twinx()
                
                offset = 60
                new_fixed_axis = par2.get_grid_helper().new_fixed_axis
                par2.axis["right"] = new_fixed_axis(loc="right",
                                                    axes=par2,
                                                    offset=(offset, 0))
            
                par2.axis["right"].toggle(all=True)
            
                #host.set_xlim(0, 2)
                #host.set_ylim(0, 2)
            
                host.set_xlabel(param_name)
                host.set_ylabel('Mean Squared Error')
                par1.set_ylabel('Node Count')
                par2.set_ylabel('Time (Milliseconds)')
                
                p1, = host.plot(param_range, train_mean,
                                color='blue', marker='o',
                                markersize=5,
                                label='Training Error')
                
                host.fill_between(param_range,
                                  train_mean + train_std,
                                  train_mean - train_std,
                                  alpha=0.15, color='blue')
                
                
                p2, = host.plot(param_range, test_mean,
                                color='green', marker='s',
                                markersize=5, linestyle='--',
                                label='Validation Error')
                
                host.fill_between(param_range,
                                  test_mean + test_std,
                                  test_mean - test_std,
                                  alpha=0.15, color='green')
                
                
                p3,  = par1.plot(param_range, nodes_mean,
                                 color='red', marker='o',
                                 markersize=5,
                                 label='Node Count')
                
                par1.fill_between(param_range,
                                  nodes_mean + nodes_std,
                                  nodes_mean - nodes_std,
                                  alpha=0.15, color='red')
                
                p4, = par2.plot(param_range, fit_time_mean,
                                color='gray', marker='3',
                                markersize=5,
                                label='Fit Time')
                
                par2.fill_between(param_range,
                                  fit_time_mean + fit_time_std,
                                  fit_time_mean - fit_time_std,
                                  alpha=0.15, color='gray')
                
                p5, = par2.plot(param_range, predict_time_mean,
                                color='orange', marker='4',
                                markersize=5,
                                label='Predict Time')
                
                par2.fill_between(param_range,
                                  predict_time_mean + predict_time_std,
                                  predict_time_mean - predict_time_std,
                                  alpha=0.15, color='orange')
                
                
                
                host.legend()

                host.axis["left"].label.set_color(p1.get_color())
                host.axis["left"].label.set_color(p2.get_color())
                par1.axis["right"].label.set_color(p3.get_color())
                par2.axis["right"].label.set_color(p4.get_color())
                par2.axis["right"].label.set_color(p5.get_color())
                
                plt.grid()
                plt.title("%s: Training, Validation Error (left)\nand Node Count/Timings (right) Versus %s" % (learner_name, param_name))
                
                if (rev_axis):
                    host.invert_xaxis()
                
                fn = save_path + dataset + '_' + learner_name + '_' + param_name + '_validation.png'
                plt.savefig(fn)
if __name__ == "__main__":
    vc = validation_curves()
    #vc.gridSearch2()
    
    dh = data_helper()
    
    X_train, X_test, y_train, y_test =  dh.load_titanic_data()
    vc.run(X_train, X_test, y_train, y_test, 'titanic')          
        
    X_train, X_test, y_train, y_test =  dh.load_wine_data()
    vc.run(X_train, X_test, y_train, y_test, 'wine')
        
        
        
        
