from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import Imputer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, Imputer

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class validation_curves:
    def __init__(self):
        pass
    
    def loadWineData(self):
        '''
        1 - fixed acidity
        2 - volatile acidity
        3 - citric acid
        4 - residual sugar
        5 - chlorides
        6 - free sulfur dioxide
        7 - total sulfur dioxide
        8 - density
        9 - pH
        10 - sulphates
        11 - alcohol
        Output variable (based on sensory data):
        12 - quality (score between 0 and 10)
        '''
        
        # load the red wine data
        # source: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/
        df = pd.read_csv('./data/winequality-red.csv', sep=';')
        
        df['phSugarRatio'] = df['pH'] / df['residual sugar']
        df['phSugarRatioScore'] = df['phSugarRatio'] / df['phSugarRatio'].std() 
        med = df['phSugarRatioScore'].median()
        abs_med = abs(med)
        df['phSugarRatioStd'] = df['phSugarRatioScore'] / abs_med
        
        #df = pd.DataFrame(np.random.rand(50, 4), columns=['a', 'b', 'c', 'd'])
        #df.plot.scatter(x='quality', y='phSugarRatioStd')
        
        # group the quality into binary good or bad
        df.loc[(df['quality'] >= 0) & (df['quality'] <= 5), 'quality'] = 0
        df.loc[(df['quality'] >= 6), 'quality'] = 1
        
        # separate the x and y data
        # y = quality, x = features (using fixed acid, volatile acid and alcohol)
        x_col_names = ['fixed acidity', 'volatile acidity', 'alcohol']
        x, y = df.loc[:,x_col_names].values, df.loc[:,'quality'].values
        
        # split the data into training and test data
        # for the wine data using 30% of the data for testing
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        
        return X_train, X_test, y_train, y_test
    
    def loadData(self):
        df = pd.read_csv('./data/shot_logs.csv', sep=',')
        
        le = LabelEncoder()
        le.fit(df['LOCATION'])
        le.transform(df['LOCATION']) 
        df['LOCATION_ENC'] = le.transform(df['LOCATION'])
            
        le = LabelEncoder()
        le.fit(df['SHOT_RESULT'])
        le.transform(df['SHOT_RESULT']) 
        df['SHOT_RESULT_ENC'] = le.transform(df['SHOT_RESULT'])
            
        x_col_names = ['player_id', 'SHOT_DIST', 'TOUCH_TIME', 'LOCATION_ENC', 'PTS_TYPE', 'DRIBBLES', 'SHOT_NUMBER', 'FINAL_MARGIN']
        x, y = df.loc[:,x_col_names].values, df.loc[:,'SHOT_RESULT_ENC'].values
        
        # split the data into training and test data
        # for the wine data using 30% of the data for testing
        X_train, X_test, y_train, y_test = train_test_split(x,
                                                            y,
                                                            test_size=0.25,
                                                            random_state=0)

        return X_train, X_test, y_train, y_test
    
    def run(self):
        
        X_train, X_test, y_train, y_test =  self.loadWineData()
        
        #kernel = 'poly'
        kernel = 'rbf'
        params_dict = {'clf__C': np.arange(1, 500, 4),
                       'clf__max_iter': np.arange(0, 1000, 25),
                       'clf__gamma': np.arange(1, 50, 1),
                       'clf__tol': np.arange(0.000001, 2.0, 0.005)}
        
        #params_dict = {'clf__degree': np.arange(0, 15, 1)}
        
        learner_name = 'SVM'
        
        for param_name in params_dict.keys():
            print(param_name)
                
            x = []
            in_sample_avg_errors = []
            std_in_sample_errors = []
            out_of_sample_avg_errors = []
            std_out_of_sample_errors = []
            avg_num_vectors = []
            std_num_vectors = []
            
            for param_value in params_dict[param_name]:
                print(param_value)
                
                clf = Pipeline([('scl', StandardScaler()),
                                ('clf', SVC(random_state=0, kernel=kernel))])
                
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
                    
                    # run the learning algorithm
                    clf.fit(X_train[train], y_train[train])
                    
                    # complexity
                    nvectors = clf.clf.n_support_.sum()
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

            fn = save_path + learner_name + '_' + param_name + '_validation.png'
            plt.savefig(fn)
            
if __name__ == "__main__":
    vc = validation_curves()
    vc.run()
              
        
        
        
        
        
