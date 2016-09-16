from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import Imputer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, Imputer
from sklearn.metrics import classification_report

import matplotlib
#matplotlib.use('Agg')
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

        pipeline = Pipeline([('scl', StandardScaler()),
                             ('clf', SVC(random_state=0))])

        parameters = {'clf__kernel': ('rbf', 'poly', 'linear'),
                      #'clf__max_iter': np.arange(0., 500., 100.),
                      'clf__gamma': np.arange(1., 50., 5.),
                      'clf__tol': np.arange(0.000001, 2.0, 0.5) }
        
        grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
        
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
            
    def gridSearch(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test =  dh.load_wine_data()
        
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

        scores = ['precision', 'recall']
        
        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()
        
            clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                               scoring='%s_weighted' % score)
            clf.fit(X_train, y_train)
        
            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            for params, mean_score, scores in clf.grid_scores_:
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean_score, scores.std() * 2, params))
            print()
        
            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            y_true, y_pred = y_test, clf.predict(X_test)
            print(classification_report(y_true, y_pred))
            print()
            
        
        
        
    def run(self, X_train, X_test, y_train, y_test, dataset):
        
        #'clf__learning_rate': {'param_value': ['constant', 'invscaling', 'adaptive'], 'reverse_xaxis': False},
        params_dict = {
                       'clf__max_iter': {'param_value': np.arange(1, 500, 10), 'reverse_xaxis': False},
                       'clf__batch_size': {'param_value': np.arange(50,500,10), 'reverse_xaxis': False},
                       'clf__learning_rate_init': {'param_value': np.arange(0.001,0.1,0.01), 'reverse_xaxis': False},
                       'clf__power_t': {'param_value': np.arange(0.01,0.1,0.01), 'reverse_xaxis': False}
        }
        
        learner_name = 'Neural Net'
        
        for param_name in params_dict.keys():
            print(param_name)
                
            x = []
            in_sample_avg_errors = []
            std_in_sample_errors = []
            out_of_sample_avg_errors = []
            std_out_of_sample_errors = []
            avg_num_vectors = []
            std_num_vectors = []
            
            for param_value in params_dict[param_name]['param_value']:
                print(param_value)
                
                clf = Pipeline([ ('scl', StandardScaler() ),
                          ('clf', MLPClassifier(random_state=0))])
                
                params = clf.get_params()
                params[param_name] = param_value
                clf.set_params(**params)
                
                # this is the 0.18dev version
                skf = StratifiedKFold(n_splits=5, random_state=0)
                
                out_sample_errors = []
                in_sample_errors = []
                num_vectors = []
                
                # do the cross validation
                for k, (train, test) in enumerate(skf.split(X=X_train, y=y_train)):
                   
                    #params = clf.get_params()
                    #print(params)
 
                    # run the learning algorithm
                    clf.fit(X_train[train], y_train[train])
                    
                    # complexity
                    nvectors = 1 #clf.named_steps['clf'].n_support_.sum()
                    num_vectors.append(nvectors)
                    
                    # in sample
                    predicted_values = clf.predict(X_train[train])
                    in_sample_mse = mean_squared_error(y_train[train], predicted_values)
                    in_sample_errors.append(in_sample_mse)
                    
                    # out of sample
                    predicted_values = clf.predict(X_train[test])
                    out_sample_mse = mean_squared_error(y_train[test], predicted_values)
                    out_sample_errors.append(out_sample_mse)

                    print('Fold:', k+1, ', Validation error: ', out_sample_mse, ', Training error: ', in_sample_mse, ', Support vectors: ', nvectors)
                   
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
                avg_vectors = np.mean(num_vectors)
                std_vectors = np.std(num_vectors)
                avg_num_vectors.append(avg_vectors)
                std_num_vectors.append(std_vectors)
                
                print('Avg validation MSE: ', avg_test_err, '+/-', test_err_std,
                      'Avg training MSE: ', avg_train_err, '+/-', train_err_std,
                      'Avg num support vectors: ', avg_vectors, '+/-', std_vectors)
                
                x.append(param_value)
                
            # prepare
            param_range = x
            train_mean = np.array(in_sample_avg_errors)
            train_std = np.array(std_in_sample_errors)
            test_mean = np.array(out_of_sample_avg_errors)
            test_std = np.array(std_out_of_sample_errors)
            vectors_mean = np.array(avg_num_vectors)
            vectors_std = np.array(std_num_vectors)
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
            
            l3 = ax2.plot(param_range, vectors_mean,
                          color='red', marker='o',
                          markersize=5,
                          label='vector count')
            
            ax2.fill_between(param_range,
                             vectors_mean + vectors_std,
                             vectors_mean - vectors_std,
                             alpha=0.15, color='red')
            
            ax1.set_xlabel(param_name)
            ax1.set_ylabel('Mean Squared Error')
            ax2.set_ylabel('Support Vector Count')

            plt.grid()
            plt.title("%s: Training, Validation Error (left)\nand Vector Count (right) Versus %s" % (learner_name, param_name))
            
            lns = l1+l2+l3
            labs = [l.get_label() for l in lns]
            ax1.legend(lns, labs, loc='center right')

            if (rev_axis):
                ax1.invert_xaxis()
            
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
              
        
        
        
        
        
