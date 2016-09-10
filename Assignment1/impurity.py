from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import Imputer, LabelEncoder
from sklearn.model_selection import train_test_split

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

boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.25,
                                                    random_state=0)

#params_dict = {"min_impurity_split": [0.2, 0.5, 0.8, 1.1, 1.4,
#                                      1.7, 2.0, 2.3, 2.6, 2.9, 3.2]}

params_dict = {"min_impurity_split": np.arange(0, 0.25, 0.005)}
params_dict = {"max_depth": np.arange(0, 5, 1)}

param_name = 'min_impurity_split'
param_name = 'max_depth'

x = []

y_num_nodes = []
y_MSE_test = []
y_MSE_train = []

# test default parameters
reg = DecisionTreeRegressor(random_state=0)
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

for param_value in params_dict[param_name]:
    reg = DecisionTreeRegressor(random_state=0,
                                min_impurity_split=param_value)
    reg.fit(X_train, y_train)

    # add number of nodes to result list
    y_num_nodes.append(reg.tree_.node_count)

    # get predictions and add to result list
    y_predicted_test = reg.predict(X_test)
    test_MSE = mean_squared_error(y_test, y_predicted_test)
    y_MSE_test.append(test_MSE)
    y_predicted_train = reg.predict(X_train)
    train_MSE = mean_squared_error(y_train, y_predicted_train)
    y_MSE_train.append(train_MSE)

    # create label for this bar
    x.append(param_value)

plt.clf()

param_range = x
train_mean = y_MSE_train
train_std = np.zeros(len(train_mean))
test_mean = y_MSE_test
test_std = np.zeros(len(train_mean))
data_label = 'a'

save_path= './output/'

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
plt.title("Validation curve: %s" % (data_label))
plt.xlabel(param_name)
plt.ylabel('Accurancy')
plt.legend(loc='lower right')

fn = save_path + data_label + '_' + param_name + '_validationcurve.png'
plt.savefig(fn)

   
# plot min_impurity_split vs Num Nodes
plt.plot(x, y_num_nodes)
# add some text for label, title and axes ticks
plt.ylabel('Number of Nodes in Tree')
plt.title('min_impurity_split vs Number of Nodes in Trees')
plt.xlabel('min_impurity_split value')
plt.tight_layout()

# plot min_impurity_split vs train_mse
plt.figure()
plt.plot(x, y_MSE_train)

plt.plot(x, y_MSE_test)

# add some text for labels, title and axes ticks
plt.ylabel('Train Set MSE')
plt.title('min_impurity_split vs Train Set MSE')
plt.xlabel('min_impurity_split value')
plt.tight_layout()

# plot min_impurity_split vs Num Nodes
plt.figure()
plt.plot(x, y_MSE_test)
# add some text for labels, title and axes ticks
plt.ylabel('Test Set MSE')
plt.title('min_impurity_split vs Test Set MSE')
plt.xlabel('min_impurity_split value')
plt.tight_layout()

plt.show()