from sklearn.preprocessing import Imputer, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

class data_helper:
    def __init__(self):
        pass
    
    def load_titanic_data(self):
        
        df = pd.read_csv('./data/titanic_train.csv', sep=',')
        
        # we need to encode sex. using the sklearn label encoder is
        # one way. however one consideration is that the learning
        # algorithm may make assumptions about the magnitude of the
        # labels. for example, male is greater than female. use
        # one hot encoder to get around this.
        #ohe = OneHotEncoder(categorical_features=[0])
        #ohe.fit_transform(x).toarray()
        
        # Even better pandas has a one hot encoding built in!
        df = pd.get_dummies(df[['Sex', 'Pclass', 'Age', 'Survived', 'Fare', 'SibSp', 'Parch']])    
    
        # this data set is missing some ages. we could impute a value
        # like the average or median. or remove the rows having missing
        # data. the disadvantage of removing values is we may be taking
        # away valuable information that the learning algorithm needs.
        imr = Imputer(strategy='most_frequent')
        imr.fit(df['Age'].reshape(-1, 1))
        imputed_data = imr.transform(df['Age'].reshape(-1, 1))
        
        df['Age']  = imputed_data
        
        
        #y = df['Survived'].values
        #x = df.iloc[:,[0,1,3,4]].values
        
        #x_col_names = df.iloc[:,[0,1,3,4]].columns
        x_col_names = ['Pclass', 'Age', 'Fare', 'SibSp', 'Parch', 'Sex_female', 'Sex_male']
        x, y = df.loc[:,x_col_names].values, df.loc[:,'Survived'].values
        
        
        # split the data into training and test data
        # for the wine data using 30% of the data for testing
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        
        return X_train, X_test, y_train, y_test
        
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
        df['phCitricAcidRatio'] = df['pH'] / df['citric acid']
        df['phVolatileAcidRatio'] = df['pH'] / df['volatile acidity']
        df['phFixedAcidRatio'] = df['pH'] / df['fixed acidity']
        
        #df['phSugarRatioScore'] = df['phSugarRatio'] / df['phSugarRatio'].std() 
        #med = df['phSugarRatioScore'].median()
        #abs_med = abs(med)
        #df['phSugarRatioStd'] = df['phSugarRatioScore'] / abs_med
        
        #df = pd.DataFrame(np.random.rand(50, 4), columns=['a', 'b', 'c', 'd'])
        #df.plot.scatter(x='quality', y='phSugarRatioStd')
        
        mean = df['quality'].mean()
        
        # group the quality into binary good or bad
        df.loc[(df['quality'] >= 0) & (df['quality'] <= mean), 'quality'] = 0
        df.loc[(df['quality'] > mean), 'quality'] = 1
        
        # separate the x and y data
        # y = quality, x = features (using fixed acid, volatile acid and alcohol)
        x_col_names = ['fixed acidity', 'volatile acidity', 'alcohol', 'total sulfur dioxide', 'sulphates', 'citric acid', 'phVolatileAcidRatio']
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
    