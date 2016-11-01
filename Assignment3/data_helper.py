from sklearn.preprocessing import Imputer, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

class data_helper:
    def __init__(self):
        pass
    
    def load_preprocessed_data(self, data_set_name):
        df = pd.read_csv('./output/titanic_test.txt', sep=',')
        y_test = df.values[:,-1]
        X_test = df.values[:,:-1]
        
        df = pd.read_csv('./output/titanic_train.txt', sep=',')
        y_train = df.values[:,-1]
        X_train = df.values[:,:-1]
        
        return X_train, X_test, y_train, y_test
        
    def pre_scale_and_export(self, X_train, X_test, y_train, y_test, data_set_name):
        '''
        This is helpful for Assignment 2 to scale and export data for use in ABAGAIL
        '''
        
        y_test.shape = (y_test.shape[0], 1)
        X_test = StandardScaler().fit_transform(X_test)
        xy_test = np.hstack((X_test, y_test))
        fn = './output/' + data_set_name + '_test.txt'
        np.savetxt(fn, xy_test, delimiter=',')
        
        y_train.shape = (y_train.shape[0], 1)
        X_train = StandardScaler().fit_transform(X_train)
        xy_train = np.hstack((X_train, y_train))
        fn = './output/' + data_set_name + '_train.txt'
        np.savetxt(fn, xy_train, delimiter=',')
        
        
    def load_preprocess_and_split_wine_data(self):
        
        '''   
        CORR
                               quality  
        fixed acidity        0.124052  
        volatile acidity     -0.390558  
        citric acid          0.226373  
        residual sugar       0.013732  
        chlorides            -0.128907  
        free sulfur dioxide  -0.050656  
        total sulfur dioxide -0.185100  
        density              -0.174919  
        pH                    -0.057731  
        sulphates             0.251397  
        alcohol               0.476166  
        quality               1.000000  
        '''
        
        df = pd.read_csv('./data/winequality-red.csv', sep=';')
        
        split = df['quality'].median()
        df['quality_2'] = df['quality']
        
        # group the quality into binary good or bad
        df.loc[(df['quality'] >= 0) & (df['quality'] < split), 'quality_2'] = 0
        df.loc[(df['quality'] >= split), 'quality_2'] = 1
        
        df['volatile_acidity_ph_ratio'] = df['volatile acidity'] / df['pH'] 
        df['fixed_acidity_ph_ratio'] = df['fixed acidity'] / df['pH'] 
        df['sulphates_residual_sugar_ratio'] = df['sulphates'] / df['residual sugar'] 
        df['alcohol_residual_sugar_ratio'] = df['alcohol'] / df['residual sugar'] 
        df['volatile_acidity_ph_ratio'] = df['volatile acidity'] / df['pH'] 


        #x_col_names = ['volatile acidity', 'alcohol', 'volatile_acidity_ph_ratio'] 
        x_col_names = ['alcohol', 'volatile acidity', 'sulphates', 'pH'] 
        
        x, y = df.loc[:,x_col_names].values, df.loc[:,'quality_2'].values
        
        # split the data into training and test data
        # for the wine data using 30% of the data for testing
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        
        return X_train, X_test, y_train, y_test
    
    
    def load_preprocess_and_split_titanic_data(self, pct_testing=0.3):
        
        '''
        Load Titanic data, preprocess with imputing, one hot encoding, and split
        
        Example row:
        
        PassengerId                          1
        Survived                             0
        Pclass                               3
        Name           Braund, Mr. Owen Harris
        Sex                               male
        Age                                 22
        SibSp                                1
        Parch                                0
        Ticket                       A/5 21171
        Fare                              7.25
        Cabin                              NaN
        Embarked                             S
        '''
        
        df = pd.read_csv('./data/titanic_train.csv', sep=',')
        
        '''
        We need to encode sex. Using the sklearn label encoder is one way. However one
        consideration is that the learning algorithm may make assumptions about the
        magnitude of the labels. For example, male is greater than female. Use one hot
        encoding to get around this.
        '''
        # Sklearn OHE
        #ohe = OneHotEncoder(categorical_features=[0])
        #ohe.fit_transform(x).toarray()
        
        # Even better pandas has a one hot encoding built in!
        dfx = pd.get_dummies(df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']])    
        
        '''
        This data set is missing some values like ages. We could impute a value like
        the average or median or remove the rows having missing data. The disadvantage
        of removing values is we may be taking away valuable information that the
        learning algorithm needs.
        '''
        
        # Impute most frequent to any missing value, except the y value
        imr = Imputer(strategy='most_frequent')
        imr.fit(dfx)
        
        # Separate x and y
        x = imr.transform(dfx)
        y = df.loc[:,'Survived'].values
        
        # Manually reduce the number of features?
        #x_col_names = ['Pclass', 'Age', 'Fare', 'SibSp', 'Parch', 'Sex_female', 'Sex_male']
        #x = df.loc[:,x_col_names].values
        
        '''
        Split the data into training and test data for the wine data holding out 30%
        by default of the data for testing
        ''' 
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=pct_testing, random_state=0)
        
        return X_train, X_test, y_train, y_test

    def load_raw_titanic_data(self, pct_testing=0.3):
        
        '''
        Load Titanic data, preprocess with imputing, one hot encoding, and split
        
        Example row:
        
        PassengerId                          1
        Survived                             0
        Pclass                               3
        Name           Braund, Mr. Owen Harris
        Sex                               male
        Age                                 22
        SibSp                                1
        Parch                                0
        Ticket                       A/5 21171
        Fare                              7.25
        Cabin                              NaN
        Embarked                             S
        '''
        
        df = pd.read_csv('./data/titanic_train.csv', sep=',')
        
        '''
        We need to encode sex. Using the sklearn label encoder is one way. However one
        consideration is that the learning algorithm may make assumptions about the
        magnitude of the labels. For example, male is greater than female. Use one hot
        encoding to get around this.
        '''
        # Sklearn OHE
        #ohe = OneHotEncoder(categorical_features=[0])
        #ohe.fit_transform(x).toarray()
        
        # Even better pandas has a one hot encoding built in!
        #df = pd.get_dummies(df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']])    
        dfx = pd.get_dummies(df[['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Age', 'Survived']])    
        
        
        '''
        This data set is missing some values like ages. We could impute a value like
        the average or median or remove the rows having missing data. The disadvantage
        of removing values is we may be taking away valuable information that the
        learning algorithm needs.
        '''
        imr = Imputer(strategy='most_frequent')
        imr.fit(dfx['Age'].reshape(-1, 1))
        imputed_data = imr.transform(dfx['Age'].reshape(-1, 1))
        
        dfx['Age']  = imputed_data
        
        
        #y = df['Survived'].values
        #x = df.iloc[:,[0,1,3,4]].values
        
        #x_col_names = df.iloc[:,[0,1,3,4]].columns
        x_col_names = ['Pclass', 'Age', 'Fare', 'SibSp', 'Parch', 'Sex_female', 'Sex_male']
        x, y = dfx.loc[:,x_col_names].values, dfx.loc[:,'Survived'].values
        
        # Impute most frequent to any missing value, except the y value
        #imr = Imputer(strategy='most_frequent')
        #imr.fit(dfx)
        
        # Separate x and y
        #x = imr.transform(dfx)
        #y = df.loc[:,'Survived'].values
        
        # Manually reduce the number of features?
        #x_col_names = ['Pclass', 'Age', 'Fare', 'SibSp', 'Parch', 'Sex_female', 'Sex_male']
        #x = df.loc[:,x_col_names].values
        
        '''
        Split the data into training and test data for the wine data holding out 30%
        by default of the data for testing
        ''' 
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=pct_testing, random_state=0)
        
        return X_train, X_test, y_train, y_test
    
if __name__== '__main__':
    dh = data_helper()
    X_train, X_test, y_train, y_test = dh.load_preprocess_and_split_titanic_data()
    dh.pre_scale_and_export(X_train, X_test, y_train, y_test, 'titanic')
    
    X_train1, X_test1, y_train1, y_test1 = dh.load_preprocessed_data('titanic')
    print(1)