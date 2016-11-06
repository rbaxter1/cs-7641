from sklearn.preprocessing import Imputer, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

class data_helper:
    def __init__(self):
        self.data_dir = 'data'
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
            
        x_col_names = ['SHOT_DIST', 'CLOSE_DEF_DIST', 'DRIBBLES']
        x, y = df.loc[:,x_col_names].values, df.loc[:,'SHOT_RESULT_ENC'].values
        
        return train_test_split(x,
                                y,
                                test_size=0.7,
                                random_state=0)

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
    
    def get_wine_data_lda_best(self):
        
        filename = './' + self.data_dir + '/wine_lda_x_train.txt'
        df1 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/wine_lda_x_test.txt'
        df2 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/wine_lda_y_train.txt'
        df3 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/wine_lda_y_test.txt'
        df4 = pd.read_csv(filename, sep=',')

        X_train = df1.values
        X_test = df2.values
        y_train = df3.values
        y_test = df4.values
        
        return X_train, X_test, y_train, y_test
    
    def get_wine_data_pca_best(self):
        
        filename = './' + self.data_dir + '/wine_pca_x_train.txt'
        df1 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/wine_pca_x_test.txt'
        df2 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/wine_pca_y_train.txt'
        df3 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/wine_pca_y_test.txt'
        df4 = pd.read_csv(filename, sep=',')

        X_train = df1.values
        X_test = df2.values
        y_train = df3.values
        y_test = df4.values
        
        return X_train, X_test, y_train, y_test
    
    def get_wine_data_rp_best(self):
        
        filename = './' + self.data_dir + '/wine_rp_x_train.txt'
        df1 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/wine_rp_x_test.txt'
        df2 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/wine_rp_y_train.txt'
        df3 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/wine_rp_y_test.txt'
        df4 = pd.read_csv(filename, sep=',')

        X_train = df1.values
        X_test = df2.values
        y_train = df3.values
        y_test = df4.values
        
        return X_train, X_test, y_train, y_test
        
    def get_wine_data_ica_best(self):
        
        filename = './' + self.data_dir + '/wine_ica_x_train.txt'
        df1 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/wine_ica_x_test.txt'
        df2 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/wine_ica_y_train.txt'
        df3 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/wine_ica_y_test.txt'
        df4 = pd.read_csv(filename, sep=',')

        X_train = df1.values
        X_test = df2.values
        y_train = df3.values
        y_test = df4.values
        
        return X_train, X_test, y_train, y_test
        
    def get_nba_data_lda_best(self):
        
        filename = './' + self.data_dir + '/nba_lda_x_train.txt'
        df1 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/nba_lda_x_test.txt'
        df2 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/nba_lda_y_train.txt'
        df3 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/nba_lda_y_test.txt'
        df4 = pd.read_csv(filename, sep=',')

        X_train = df1.values
        X_test = df2.values
        y_train = df3.values
        y_test = df4.values
        
        return X_train, X_test, y_train, y_test
    
    def get_nba_data_pca_best(self):
        
        filename = './' + self.data_dir + '/nba_pca_x_train.txt'
        df1 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/nba_pca_x_test.txt'
        df2 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/nba_pca_y_train.txt'
        df3 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/nba_pca_y_test.txt'
        df4 = pd.read_csv(filename, sep=',')

        X_train = df1.values
        X_test = df2.values
        y_train = df3.values
        y_test = df4.values
        
        return X_train, X_test, y_train, y_test
    
    def get_nba_data_rp_best(self):
        
        filename = './' + self.data_dir + '/nba_rp_x_train.txt'
        df1 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/nba_rp_x_test.txt'
        df2 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/nba_rp_y_train.txt'
        df3 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/nba_rp_y_test.txt'
        df4 = pd.read_csv(filename, sep=',')

        X_train = df1.values
        X_test = df2.values
        y_train = df3.values
        y_test = df4.values
        
        return X_train, X_test, y_train, y_test
        
    def get_nba_data_ica_best(self):
        
        filename = './' + self.data_dir + '/nba_ica_x_train.txt'
        df1 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/nba_ica_x_test.txt'
        df2 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/nba_ica_y_train.txt'
        df3 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/nba_ica_y_test.txt'
        df4 = pd.read_csv(filename, sep=',')

        X_train = df1.values
        X_test = df2.values
        y_train = df3.values
        y_test = df4.values
        
        return X_train, X_test, y_train, y_test
        
if __name__== '__main__':
    print(1)
