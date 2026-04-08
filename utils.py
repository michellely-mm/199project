import pandas as pd
import numpy as np
import os
from pathlib import Path
from statsmodels.tsa.seasonal import STL

#Path of the data
DATA_PATH = Path.home()/"199project"/"data"/"Yearly data"

#20 ETF list
ETF_LIST = [
    'SPY', 'QQQ', 'DIA', 'MDY',
    'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLP', 'XLY', 'XLB', 'XLU',
    'EWJ', 'EWG', 'EWU', 'EWQ', 'EWC', 'EWH', 'EWA'
]

def load_one_day(date_int):
    """
    load one day data 
    date_int: integer; eg. 20000103
    """
    year = str(date_int)[:4]
    filename = f"{date_int}.csv.gz"
    filepath = DATA_PATH/year/filename 

    if not filepath.exists():
        return None
    
    df = pd.read_csv(filepath, compression = 'gzip')
    return df

def load_date_range(start_date, end_date):
    """
    load all data in the given range
    start_date, end_date: integers; eg. 20000103, 20001231
    """
    calendar_path = Path.home()/"199project"/"data"/"nyse_business_days_1990_2026.csv"
    calendar = pd.read_csv(calendar_path, header = None, names = ["date"]) #name this column = date

    dates = calendar[(calendar["date"] >= start_date) &
                     (calendar["date"] <= end_date)]["date"].tolist()
    
    all_days = []
    for date in dates:
        df = load_one_day(date)
        if df is not None and len(df) > 0:
            all_days.append(df)
    #returned a big DF: stacking all dates corresponding with all the stocks together
    return pd.concat(all_days, ignore_index=True)

#3 times 2 = 6 series
#Each series do STL (2 features) = 12 features per ETF
#20 ETFs * 12 = 240 features
#20 ETFs * 1 = 20 volume and 20 volatility features
#ETF = 280 features
def extract_etf_returns(df):
    """
    extract ETF return, volume, volatility from daily data
    RR = Raw Return
    MR = Market-Adjusted Return
    """
    etf_df = df[df['ticker'].isin(ETF_LIST)][
        ['date', 'ticker', 'OPCL', 'pvCLCL', 'fret_RR_CLOP', 
         'volume', 'high', 'low', 'close']
    ].copy()

    #renaming
    etf_df = etf_df.rename(columns= {
        'OPCL': 'RR_OPCL',
        'pvCLCL': 'RR_CLCL',
        'fret_RR_CLOP': 'RR_CLOP'
    })

    #calculate for volatility
    etf_df['volMM'] = (etf_df['high'] - etf_df['low']).abs() / etf_df['close'].abs() * 10000

    return etf_df

def add_market_adjusted_returns(etf_df):
    """
    calculate Market-Adjusted Return (MR)
    MR = RR - SPY RR
    """
    # use daily spy return to be the standard
    spy_returns = etf_df[etf_df['ticker'] == 'SPY'][
        ['date', 'RR_OPCL', 'RR_CLCL', 'RR_CLOP']
    ].rename(columns={
        'RR_OPCL': 'SPY_OPCL',
        'RR_CLCL': 'SPY_CLCL',
        'RR_CLOP': 'SPY_CLOP'
    })

    # merge SPY return
    etf_df = etf_df.merge(spy_returns, on='date', how='left')

    # calculate MR
    etf_df['MR_OPCL'] = etf_df['RR_OPCL'] - etf_df['SPY_OPCL']
    etf_df['MR_CLCL'] = etf_df['RR_CLCL'] - etf_df['SPY_CLCL']
    etf_df['MR_CLOP'] = etf_df['RR_CLOP'] - etf_df['SPY_CLOP']

    etf_df = etf_df.drop(columns=['SPY_OPCL', 'SPY_CLCL', 'SPY_CLOP'])

    return etf_df

#6 series that need to do return series
RETURN_COLS = ['RR_OPCL', 'RR_CLCL', 'RR_CLOP', 'MR_OPCL', 'MR_CLCL', 'MR_CLOP']

def compute_stl_features(etf_series, period = 5):
    """
    For every ETF's return series do STL decomposition
    etf_series: length = 252 times sequence (we use 252 data to do STL decomposition)
    period: seasonal period (5 trading days per week)
    """
    if len(etf_series) < 2 * period:
        return None, None
    stl = STL(etf_series, period = period, robust = True)
    result = stl.fit() #STL into trend, seasonal, and residual

    #just return trend and seaonal
    trend = result.trend.iloc[-1] #we just need the last row whihc is the data at time t
    seasonal = result.seasonal.iloc[-1]
    return trend, seasonal

def build_etf_stl_features(etf_df):
    """
    For each ETF's 6 return series do STL
    return a dict, key is date, and value is 240 features (6*2 = 12; 12*20 = 240)
    """
    #series we are doing STL on
    return_cols = ['RR_OPCL', 'RR_CLCL', 'RR_CLOP', 
                   'MR_OPCL', 'MR_CLCL', 'MR_CLOP']
    features = {}
    for etf in ETF_LIST:
        #only keep the row with tick = etf in the list; set date becomes the index
        etf_data = etf_df[etf_df['ticker'] == etf].set_index('date')
        for col in return_cols:
            if col not in etf_data.columns:
                continue
            #get this column to do STL decomposition
            series = etf_data[col]
            trend, seasonal = compute_stl_features(series)

            if trend is None:
                continue

            features[f'{etf}_{col}_trend'] = trend
            features[f'{etf}_{col}_seasonal'] = seasonal

    return features

def build_etf_volume_vol_features(etf_df):
    """
    Get volume and volMM data of the present day
    20 ETFs * 2 = 40 features 
    """
    features = {}

    #abtain the last day data
    last_date = etf_df['date'].max()
    last_day = etf_df[etf_df['date'] == last_date]

    for etf in ETF_LIST:
        etf_row = last_day[last_day['ticker'] == etf]
        if len(etf_row) == 0:
            continue
        features[f'{etf}_volume'] = etf_row['volume'].values[0]
        features[f'{etf}_volMM'] = etf_row['volMM'].values[0]
    
    return features

def build_stock_features(df, ticker, date):
    """
    build daily specific stock features
    df: load_data_range return enough historical dataframe
    ticker: stock id
    date: current date; eg. 20000103
    """
    #rank all the stocks 
    stock_df = df[df['ticker'] == ticker].sort_values('date')

    #find the current day index position
    today_idx = stock_df[stock_df['date'] == date].index
    if len(today_idx) ==  0:
        return None
    
    #turn it into series: 1D, only have index and value
    today = stock_df[stock_df['date'] == date].iloc[0]

    #Group 1: same day return
    features = {
        'RR_OPCL': today['OPCL'],
        'RR_CLCL': today['pvCLCL'],
        'RR_CLOP': today['fret_RR_CLOP'],
        'MR_OPCL': today['OPCL'] - today['SPpvCLCL'],
        'MR_CLCL': today['pvCLCL'] - today['SPpvCLCL'],
        'MR_CLOP': today['fret_RR_CLOP'] - today['SPpvCLCL'],
    }

    #Group 2: past 5 days cumulated return
    past5 = stock_df[stock_df['date'] < date].tail(5)

    features['cum_RR_OPCL'] = past5['OPCL'].sum()
    features['cum_RR_CLCL'] = past5['pvCLCL'].sum()
    features['cum_RR_CLOP'] = past5['fret_RR_CLOP'].sum()
    features['cum_MR_OPCL'] = (past5['OPCL'] - past5['SPpvCLCL']).sum()
    features['cum_MR_CLCL'] = (past5['pvCLCL'] - past5['SPpvCLCL']).sum()
    features['cum_MR_CLOP'] = (past5['fret_RR_CLOP'] - past5['SPpvCLCL']).sum()
    
    return features
    
