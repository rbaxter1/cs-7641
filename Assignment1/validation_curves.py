#from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import Imputer, LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import StratifiedKFold

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('./data/shot_logs.csv', sep=',')
    
#x_col_names = ['player_id', 'SHOT_DIST', 'TOUCH_TIME']
#df = pd.get_dummies(df[['SHOT_RESULT', 'LOCATION', 'Age', 'Survived']])


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
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

#boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.25,
                                                    random_state=0)

#params_dict = {"min_impurity_split": [0.2, 0.5, 0.8, 1.1, 1.4,
#                                      1.7, 2.0, 2.3, 2.6, 2.9, 3.2]}


params_dict = {'min_impurity_split': np.arange(0, 0.5, 0.01),
               'max_depth': np.arange(1, 100, 1),
               'min_samples_split': np.arange(1, 50, 1),
               'min_samples_leaf': np.arange(2, 100, 1),
               'min_weight_fraction_leaf': np.arange(0.0, 0.5, 0.05),
               'max_leaf_nodes': np.arange(2, 50, 1)}
#params_dict = {'min_impurity_split': np.arange(0, 0.5, 0.01)}

#param_name = 'max_depth'
#param_name = 'min_impurity_split'

# test default parameters
'''
reg = DecisionTreeClassifier(random_state=0)
reg.fit(X_train, y_train)

# add num nodes to result list
y_num_nodes.append(reg.tree_.node_count)

# get predictions and add to result list
y_predicted_test = reg.predict(X_test)
test_MSE = mean_squared_error(y_test, y_predicted_test)
y_MSE_test.append(test_MSE)
y_predicted_train = reg.predict(X_train)
train_MSE = mean_squared_error(y_train, y_predicted_train)
y_MSE_train.append(train_MSE)

# add a label for default case
x.append(0)
'''

for param_name in params_dict.keys():
    print(param_name)
        
    x = []
    y_num_nodes = []
    y_num_nodes_sd = []
    y_MSE_test = []
    y_MSE_train = []
    y_MSE_test_sd = []
    y_MSE_train_sd = []
    
    for param_value in params_dict[param_name]:
        print(param_value)
        
        clf = DecisionTreeClassifier(criterion='entropy', random_state=0)
        params = clf.get_params()
        params[param_name] = param_value
        clf.set_params(**params)
        
        # this is the 0.18dev version
        skf = StratifiedKFold(n_splits=5, random_state=0)
        
        # do the cross validation
        train_scores = []
        test_scores = []
        train_scores_sd = []
        test_scores_sd = []
        num_nodes = []
        num_nodes_inner = []
        
        for k, (train, test) in enumerate(skf.split(X=X_train, y=y_train)):
            
            # run the learning algorithm
            clf.fit(X_train[train], y_train[train])
            
            num_nodes_inner = clf.tree_.node_count
            
            y_predicted_train = clf.predict(X_train[train])
            # should be the same
            train_MSE = mean_squared_error(y_train[train], y_predicted_train)
            #train_MSE2 = 1. - clf.score(X_train[test], y_train[test])
            train_scores.append(train_MSE)
            
            #test_score = clf.score(X_test, y_test)
            #test_scores.append(test_score)
            print('Fold:', k+1, ', Training MSE:', train_MSE, ', Nodes: ', num_nodes_inner)
        
            y_predicted_test = clf.predict(X_train[test])
            test_MSE = mean_squared_error(y_train[test], y_predicted_test)
            #test_MSE2 = 1. - clf.score(X_test, y_test)
            test_scores.append(test_MSE)
            print('Test MSE:', test_MSE)
            
        
        train_score = np.mean(train_scores)
        train_score_sd = np.std(train_scores)
        y_MSE_train.append(train_score)
        y_MSE_train_sd.append(train_score_sd)
        print('Training MSE is', train_score)
        
        test_score = np.mean(test_scores)
        test_score_sd = np.std(test_scores)
        y_MSE_test.append(test_score)
        y_MSE_test_sd.append(test_score_sd)
        print('Test MSE is', test_score)
        
        num_nodes = np.mean(num_nodes_inner)
        num_nodes_sd = np.std(num_nodes_inner)
        y_num_nodes.append(num_nodes)
        y_num_nodes_sd.append(num_nodes_sd)
        
        x.append(param_value)
        
    plt.cla()    
    plt.clf()
    
    param_range = x
    train_mean = np.array(y_MSE_train)
    train_std = np.array(y_MSE_train_sd)
    test_mean = np.array(y_MSE_test)
    test_std = np.array(y_MSE_test_sd)
    data_label = 'a'
    
    save_path= './output/'
    
    plt.plot(param_range, train_mean,
                color='blue', marker='o',
                markersize=5,
                label='training error')
    
    plt.fill_between(param_range,
                     train_mean + train_std,
                     train_mean - train_std,
                     alpha=0.15, color='blue')
    
    plt.plot(param_range, test_mean,
             color='green', marker='s',
             markersize=5, linestyle='--',
             label='validation error')
    
    plt.fill_between(param_range,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')
    
    plt.grid()
    plt.title("Validation curve: %s" % (data_label))
    plt.xlabel(param_name)
    plt.ylabel('Mean Squared Error')
    plt.legend(loc='lower right')
    
    fn = save_path + data_label + '_' + param_name + '_validationcurve.png'
    plt.savefig(fn)
    
    plt.cla()
    plt.clf()
    
    param_range = x
    nodes_mean = np.array(y_num_nodes)
    nodes_std = np.array(y_num_nodes_sd)
    data_label = 'a'
    
    save_path= './output/'
    
    plt.plot(param_range, nodes_mean,
                color='blue', marker='o',
                markersize=5,
                label='node count')
    
    plt.fill_between(param_range,
                     nodes_mean + nodes_std,
                     nodes_mean - nodes_std,
                     alpha=0.15, color='blue')
    
    plt.grid()
    plt.title("Validation curve: %s" % (data_label))
    plt.xlabel(param_name)
    plt.ylabel('Node Count')
    plt.legend(loc='lower right')
    
    fn = save_path + data_label + '_' + param_name + 'nodes_validationcurve.png'
    plt.savefig(fn)
        
        
        
        
        
        
        
        
        
