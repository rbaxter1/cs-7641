import pandas as pd
import urllib as u
from bs4 import BeautifulSoup as bs
import warnings
warnings.filterwarnings("ignore")

import quandl

def _get_quarterly_earnings(url):
    try:
        html_source = u.request.urlopen(url).read()
        soup = bs(html_source, 'lxml')
        # 
        # table
        table = soup.find_all('table', attrs={'class': 'yfnc_tableout1'})
        header = [th.text for th in table[0].find_all(class_='yfnc_tablehead1')]
        header_title = header[0]
        header_cols = header[1:5]
        index_row_labels = header[-5:]
        body = [[td.text for td in row.select('td')] for row in table[0].find_all('tr')]
        body = body[1:]
        df = pd.DataFrame.from_records(body)
        df = df.ix[:, 1:]
        df.index = index_row_labels
        header_cols = pd.Series(header_cols)
        header_cols = header_cols.str.replace(
            'Year', 'Year ').str.replace('Qtr.', 'Qtr. ')
        df.columns = header_cols
        eps_est = df.convert_objects(convert_numeric=True)
    except Exception as e:
        print(e)
    return eps_est

def _get_eps_estimates(url):
    try:
        html_source = u.request.urlopen(url).read()
        soup = bs(html_source, 'lxml')
        # 
        # table
        table = soup.find_all('table', attrs={'class': 'yfnc_tableout1'})
        header = [th.text for th in table[0].find_all(class_='yfnc_tablehead1')]
        header_title = header[0]
        header_cols = header[1:5]
        index_row_labels = header[-5:]
        body = [[td.text for td in row.select('td')] for row in table[0].find_all('tr')]
        body = body[1:]
        df = pd.DataFrame.from_records(body)
        df = df.ix[:, 1:]
        df.index = index_row_labels
        header_cols = pd.Series(header_cols)
        header_cols = header_cols.str.replace(
            'Year', 'Year ').str.replace('Qtr.', 'Qtr. ')
        df.columns = header_cols
        eps_est = df.convert_objects(convert_numeric=True)
    except Exception as e:
        print(e)
    return eps_est



if __name__ == "__main__":
    
    
    quandl.ApiConfig.api_key = 'g1CzojR2ySKV9VuewTyV'
    data = quandl.get_table("SF1", ticker="MSFT")


    symbol = 'AAPL'
    
    base_url = r'https://eresearch.fidelity.com/eresearch/evaluate/fundamentals/earnings.jhtml?tab=details&symbols='.format(symbol)
    quarterly_earnings = _get_quarterly_earnings(base_url)
    quarterly_earnings
    #eps_est = _get_eps_estimates(base_url)
    #eps_est
