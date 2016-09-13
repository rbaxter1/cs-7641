from sklearn.preprocessing import Imputer, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

class data_helper:
    def __init__(self):
        pass
    
    def load_wine_data(self):
        
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
        x_col_names = ['fixed acidity', 'volatile acidity', 'alcohol', 'total sulfur dioxide', 'sulphates']
        x, y = df.loc[:,x_col_names].values, df.loc[:,'quality'].values
        
        # split the data into training and test data
        # for the wine data using 30% of the data for testing
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        
        return X_train, X_test, y_train, y_test
    
    def load_nba_data(self):
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
    