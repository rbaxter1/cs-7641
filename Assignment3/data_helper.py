from sklearn.preprocessing import Imputer, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

class data_helper:
    def __init__(self):
        pass
    
    def get_nba_data(self):
        df = pd.read_csv('./data/shot_logs.csv', sep=',')
        
        le = LabelEncoder()
        le.fit(df['LOCATION'])
        le.transform(df['LOCATION']) 
        df['LOCATION_ENC'] = le.transform(df['LOCATION'])
            
        le = LabelEncoder()
        le.fit(df['SHOT_RESULT'])
        le.transform(df['SHOT_RESULT']) 
        df['SHOT_RESULT_ENC'] = le.transform(df['SHOT_RESULT'])
            
        x_col_names = ['SHOT_DIST', 'TOUCH_TIME', 'LOCATION_ENC', 'PTS_TYPE', 'DRIBBLES', 'FINAL_MARGIN']
        x_col_names = ['SHOT_DIST', 'CLOSE_DEF_DIST', 'DRIBBLES']
        x, y = df.loc[:,x_col_names].values, df.loc[:,'SHOT_RESULT_ENC'].values
        
        return train_test_split(x,
                                y,
                                test_size=0.7,
                                random_state=0)

    
    def get_wine_data_all(self):
        
        df = pd.read_csv('./data/winequality-red.csv', sep=';')
        
        split = df['quality'].median()
        df['quality_2'] = df['quality']
        
        # group the quality into binary good or bad
        df.loc[(df['quality'] >= 0) & (df['quality'] < split), 'quality_2'] = 0
        df.loc[(df['quality'] >= split), 'quality_2'] = 1
        
        df.dropna(how='all', inplace=True)
        
        x_col_names = ['fixed acidity', 'citric acid', 'alcohol', 'residual sugar', 'chlorides', 'volatile acidity', 'sulphates', 'pH'] 
    
        y = df.loc[:,'quality_2'].values
        df = df.drop('quality', 1)
        df = df.drop('quality_2', 1)
        x = df.values
        
        return train_test_split(x, y, test_size=0.3, random_state=0)
                
    def get_wine_data(self):
        
        df = pd.read_csv('./data/winequality-red.csv', sep=';')
        
        split = df['quality'].median()
        df['quality_2'] = df['quality']
        
        # group the quality into binary good or bad
        df.loc[(df['quality'] >= 0) & (df['quality'] < split), 'quality_2'] = 0
        df.loc[(df['quality'] >= split), 'quality_2'] = 1
        
        df.dropna(how='all', inplace=True)
        
        x_col_names = ['alcohol', 'volatile acidity', 'sulphates', 'pH'] 
        
        x = df.loc[:,x_col_names].values
        y = df.loc[:,'quality_2'].values
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        y_train = y_train.reshape(-1, 1).astype(float)
        y_test = y_test.reshape(-1, 1).astype(float)
        
        return X_train, X_test, y_train, y_test
    
        
if __name__== '__main__':
    dh = data_helper()
    X_train, X_test, y_train, y_test = dh.load_nba_data()
    
    X_train, X_test, y_train, y_test = dh.load_preprocess_and_split_titanic_data()
    dh.pre_scale_and_export(X_train, X_test, y_train, y_test, 'titanic')
    
    X_train1, X_test1, y_train1, y_test1 = dh.load_preprocessed_data('titanic')
    print(1)
