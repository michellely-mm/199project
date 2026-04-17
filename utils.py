import pandas as pd
import numpy as np
import os
from pathlib import Path
from statsmodels.tsa.seasonal import STL

# Path of the data
DATA_PATH = Path.home() / "199project" / "data" / "Yearly data"

FEATURES_PATH = Path.home() / "199project" / "features"
FEATURES_PATH.mkdir(exist_ok=True)

import pickle

def save_etf_cache(cache: dict, path=None):
    if path is None:
        path = FEATURES_PATH / "etf_stl_cache.pkl"
    with open(path, 'wb') as f:
        pickle.dump(cache, f)
    print(f"Saved to {path}")

def load_etf_cache() -> dict:
    path = FEATURES_PATH / "etf_stl_cache.pkl"
    with open(path, 'rb') as f:
        return pickle.load(f)

# 20 ETF list
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
    filepath = DATA_PATH / year / filename

    if not filepath.exists():
        return None

    df = pd.read_csv(filepath, compression='gzip')
    return df

def load_date_range(start_date, end_date, tickers: list[str] | None = None) -> pd.DataFrame:
    """
    load all data in the given range
    start_date, end_date: integers; eg. 20000103, 20001231
    """
    calendar_path = Path.home() / "199project" / "data" / "nyse_business_days_1990_2026.csv"
    calendar = pd.read_csv(calendar_path, header=None, names=["date"])

    dates = calendar[(calendar["date"] >= start_date) &
                     (calendar["date"] <= end_date)]["date"].tolist()

    all_days = []
    for date in dates:
        df = load_one_day(date)
        if df is None or len(df) == 0:
            continue
        if tickers is not None:
            if "ticker" not in df.columns:
                continue
            # only load the stocks needed, not all stocks
            df = df[df["ticker"].isin(tickers)]
        if not df.empty:
            all_days.append(df)

    # returned a big DF: stacking all dates corresponding with all the stocks together
    return pd.concat(all_days, ignore_index=True)

# Feature dimensions:
# ETF STL features:    20 ETFs * 6 series * 2 (trend + seasonal) = 240
# ETF vol features:    20 ETFs * 2 (volume + volMM)              =  40
# Stock features:      6 same-day + 6 cumulative                 =  12
# Total                                                          = 292

def extract_etf_returns(df):
    """
    extract ETF return, volume, volatility from daily data
    RR = Raw Return
    MR = Market-Adjusted Return
    
    Note: fret_RR_CLOP (tomorrow open - today close) is excluded
    because it contains t+1 information and would cause data leakage.
    OPCL, pvCLCL, and CLOP are kept. CLOP is derived via the identity
    (1 + CLCL) / (1 + OPCL) - 1, using only same-day observable data.
    fret_RR_CLOP (tomorrow open - today close) is excluded due to leakage.
    """
    etf_df = df[df['ticker'].isin(ETF_LIST)][
        ['date', 'ticker', 'OPCL', 'pvCLCL',
         'volume', 'high', 'low', 'close']
    ].copy()

    # renaming: keep only same-day observable returns
    etf_df = etf_df.rename(columns={
        'OPCL':   'RR_OPCL',
        'pvCLCL': 'RR_CLCL',
    })
    etf_df['RR_CLOP'] = (1 + etf_df['RR_CLCL']) / (1 + etf_df['RR_OPCL']) - 1
    # calculate intraday volatility proxy
    etf_df['volMM'] = (etf_df['high'] - etf_df['low']).abs() / etf_df['close'].abs() * 10000

    return etf_df

def add_market_adjusted_returns(etf_df):
    """
    calculate Market-Adjusted Return (MR)
    MR = RR - SPY RR
    Applied to OPCL, CLCL, and CLOP series.
    Note: the excluded series is fret_RR_CLOP
    """
    # use daily SPY return as the market benchmark
    spy_returns = etf_df[etf_df['ticker'] == 'SPY'][
        ['date', 'RR_OPCL', 'RR_CLCL', 'RR_CLOP']
    ].rename(columns={
        'RR_OPCL': 'SPY_OPCL',
        'RR_CLCL': 'SPY_CLCL',
        'RR_CLOP': 'SPY_CLOP',
    })

    # merge SPY return
    etf_df = etf_df.merge(spy_returns, on='date', how='left')

    # calculate MR = RR - SPY RR
    etf_df['MR_OPCL'] = etf_df['RR_OPCL'] - etf_df['SPY_OPCL']
    etf_df['MR_CLCL'] = etf_df['RR_CLCL'] - etf_df['SPY_CLCL']
    etf_df['MR_CLOP'] = etf_df['RR_CLOP'] - etf_df['SPY_CLOP']

    etf_df = etf_df.drop(columns=['SPY_OPCL', 'SPY_CLCL','SPY_CLOP'])

    return etf_df

RETURN_COLS = ['RR_OPCL', 'RR_CLCL', 'RR_CLOP', 'MR_OPCL', 'MR_CLCL', 'MR_CLOP']

def compute_stl_features(etf_series, period=5):
    """
    For every ETF's return series do STL decomposition
    etf_series: length = 252 times sequence (we use 252 data to do STL decomposition)
    period: seasonal period (5 trading days per week)
    """
    if len(etf_series) < 2 * period:
        return None, None
    stl = STL(etf_series, period=period, robust=True)
    result = stl.fit()  # STL into trend, seasonal, and residual

    # just return trend and seasonal at time t (last value)
    trend    = result.trend.iloc[-1]
    seasonal = result.seasonal.iloc[-1]
    return trend, seasonal

def build_etf_volume_vol_features(etf_df):
    """
    Get volume and volMM data of the present day
    20 ETFs * 2 = 40 features 
    """
    features = {}
    last_date = etf_df['date'].max()
    last_day  = etf_df[etf_df['date'] == last_date]

    for etf in ETF_LIST:
        etf_row = last_day[last_day['ticker'] == etf]
        if len(etf_row) == 0:
            features[f'{etf}_volume'] = np.nan
            features[f'{etf}_volMM']  = np.nan
            continue
        features[f'{etf}_volume'] = etf_row['volume'].values[0]
        features[f'{etf}_volMM']  = etf_row['volMM'].values[0]

    return features

def build_stock_features(df, ticker, date):
    """
    build daily stock-specific features
    df: dataframe returned by load_date_range with enough historical data
    ticker: stock identifier
    date: current date as integer; eg. 20000103

    Features included (12 total):
      Group 1 - same-day observable returns (6):
        RR_OPCL, RR_CLCL, RR_CLOP, MR_OPCL, MR_CLCL, MR_CLOP
      Group 2 - past 5 days cumulative returns (6):
        cum_RR_OPCL, cum_RR_CLCL, cum_RR_CLOP,
        cum_MR_OPCL, cum_MR_CLCL, cum_MR_CLOP

    Note: CLOP_t = (open_t - close_{t-1}) / close_{t-1} is computed via
    the identity (1 + CLCL) / (1 + OPCL) - 1, using only day-t data.
    fret_RR_CLOP (tomorrow open - today close) is NOT used anywhere.
    """
    stock_df = df[df['ticker'] == ticker].sort_values('date')

    if stock_df[stock_df['date'] == date].empty:
        return None

    today = stock_df[stock_df['date'] == date].iloc[0]

    # ── 个股当天三个 return ───────────────────────────────────────────────
    stock_opcl = today['OPCL']
    stock_clcl = today['pvCLCL']
    stock_clop = (1 + stock_clcl) / (1 + stock_opcl) - 1

    # ── SPY 当天，用于 MR 基准 ────────────────────────────────────────────
    spy_today = df[(df['ticker'] == 'SPY') & (df['date'] == date)]
    if spy_today.empty:
        spy_opcl = np.nan
        spy_clcl = np.nan
        spy_clop = np.nan
    else:
        s = spy_today.iloc[0]
        spy_opcl = s['OPCL']
        spy_clcl = s['pvCLCL']
        spy_clop = (1 + spy_clcl) / (1 + spy_opcl) - 1

    # ── Group 1: same-day (6 个特征) ─────────────────────────────────────
    features = {
        'RR_OPCL': stock_opcl,
        'RR_CLCL': stock_clcl,
        'RR_CLOP': stock_clop,
        'MR_OPCL': stock_opcl - spy_opcl,
        'MR_CLCL': stock_clcl - spy_clcl,
        'MR_CLOP': stock_clop - spy_clop,
    }

    # ── Group 2: past 5 days cumulative return (6 个特征) ────────────────
    past5 = stock_df[stock_df['date'] < date].tail(5)

    # 个股过去5天累计
    p5_opcl = past5['OPCL'].sum()
    p5_clcl = past5['pvCLCL'].sum()
    p5_clop = past5.apply(
        lambda r: (1 + r['pvCLCL']) / (1 + r['OPCL']) - 1, axis=1
    ).sum()

    # SPY 过去5天累计
    spy_past5 = df[
        (df['ticker'] == 'SPY') &
        (df['date'].isin(past5['date'].tolist()))
    ].set_index('date').reindex(past5['date'].tolist())

    spy_p5_opcl = spy_past5['OPCL'].fillna(0).sum()
    spy_p5_clcl = spy_past5['pvCLCL'].fillna(0).sum()
    spy_p5_clop = spy_past5.apply(
        lambda r: (1 + r['pvCLCL']) / (1 + r['OPCL']) - 1
        if pd.notna(r['OPCL']) else 0.0,
        axis=1
    ).fillna(0).sum()

    features['cum_RR_OPCL'] = p5_opcl
    features['cum_RR_CLCL'] = p5_clcl
    features['cum_RR_CLOP'] = p5_clop
    features['cum_MR_OPCL'] = p5_opcl - spy_p5_opcl
    features['cum_MR_CLCL'] = p5_clcl - spy_p5_clcl
    features['cum_MR_CLOP'] = p5_clop - spy_p5_clop

    return features


def load_calendar() -> list[int]:
    cal = pd.read_csv(
        Path.home() / "199project" / "data" / "nyse_business_days_1990_2026.csv",
        header=None, names=["date"]
    )
    return sorted(cal["date"].tolist())

# Rolling window design:
# First tau  = end of 2015
# Training   : ~2012 to 2015 (1000 days)
# Validation : ~2015 - end of 2015 (252 days to pick lambda)
# Last tau   = end of 2020
# Retrain model every half year (June and December)

def get_retraining_dates(calendar: list[int],
                         first_tau: int,
                         last_tau: int) -> list[int]:
    """
    return all retraining dates tau
    For every tau: retrain model after this day, use data before this date
    to build training/validation window, then predict 126 days forward.

    Example:
        Tau1 = 20151231 → predict 20160104 to 20160630
        Tau2 = 20160630 → predict 20160701 to 20161230
    """
    df = pd.DataFrame({"date": calendar})
    df["ym"] = df["date"].astype(str).str[:6]  # eg. 201506

    # keep only June and December month-ends
    df = df[df["ym"].str[4:].isin(["06", "12"])]

    # find the last trading date of each month
    month_ends = df.groupby("ym")["date"].max().tolist()

    # filter the range
    return [d for d in month_ends if first_tau <= d <= last_tau]

def get_window_bounds(calendar: list[int], tau: int) -> dict:
    if tau not in calendar:
        raise ValueError(f"tau={tau} not found in calendar.")
    
    idx = calendar.index(tau)
    
    if idx < 1252:
        raise ValueError(
            f"tau={tau} at index {idx}: need at least 1252 prior "
            f"trading days. Earliest valid tau = {calendar[1252]}."
        )
    if idx + 126 >= len(calendar):
        raise ValueError(
            f"tau={tau} at index {idx}: need at least 126 future "
            f"trading days. Latest valid tau = {calendar[-(127)]}."
        )
    
    return {
        "train_start": calendar[idx - 1251],
        "train_end":   calendar[idx - 252],
        "val_start":   calendar[idx - 251],
        "val_end":     tau,
        "pred_start":  calendar[idx + 1],
        "pred_end":    calendar[idx + 126],
    }

def build_etf_stl_features(etf_panel: pd.DataFrame,
                            month_end_date: int,
                            all_dates: list[int]) -> dict:
    """
    For the end of every month, calculate all ETFs' STL features (160 total)
    20 ETFs * 6 return series * 2 (trend + seasonal) = 240 features

    etf_panel     : ETF dataframe including MR columns
    month_end_date: STL window ends at this date (inclusive)
    all_dates     : trading calendar

    Only uses data up to and including month_end_date (no leakage).
    """
    # find 252-day window ending at month_end_date
    dates_up_to  = [d for d in all_dates if d <= month_end_date]
    window_dates = dates_up_to[-252:]

    features = {}

    for etf in ETF_LIST:
        etf_data = (etf_panel[etf_panel['ticker'] == etf]
                    .set_index('date'))

        for col in RETURN_COLS:
            key_trend    = f'{etf}_{col}_trend'
            key_seasonal = f'{etf}_{col}_seasonal'

            if col not in etf_data.columns:
                features[key_trend]    = np.nan
                features[key_seasonal] = np.nan
                continue

            series = etf_data.loc[
                etf_data.index.isin(window_dates), col
            ].dropna()

            if len(series) < 10:
                features[key_trend]    = np.nan
                features[key_seasonal] = np.nan
                continue

            try:
                res = STL(series, period=5, robust=True).fit()
                features[key_trend]    = float(res.trend.iloc[-1])
                features[key_seasonal] = float(res.seasonal.iloc[-1])
            except Exception:
                features[key_trend]    = np.nan
                features[key_seasonal] = np.nan

    return features

def build_etf_feature_cache(etf_panel: pd.DataFrame,
                             all_dates: list[int],
                             start_date: int,
                             end_date: int) -> dict:
    """
    For the end of every month from start_date to end_date,
    calculate ETF STL features and save as {month_end_date: feature_dict}
    return {month_end_date: feature_dict}
    """
    df = pd.DataFrame({"date": [d for d in all_dates
                                 if start_date <= d <= end_date]})
    df["ym"] = df["date"].astype(str).str[:6]
    month_ends = df.groupby("ym")["date"].max().tolist()

    cache = {}
    for i, me in enumerate(month_ends):
        print(f"[{i+1}/{len(month_ends)}] Computing STL for {me}...")
        cache[me] = build_etf_stl_features(etf_panel, me, all_dates)

    return cache

import bisect

def get_etf_features(cache: dict, month_ends: list[int], 
                     date: int,
                     calendar: list[int] = None,
                     max_stale_days: int = 63) -> dict:
    idx = bisect.bisect_right(month_ends, date) - 1
    
    if idx < 0:
        raise ValueError(
            f"date={date} is earlier than the first cached "
            f"month-end {month_ends[0]}."
        )
    
    nearest_month_end = month_ends[idx]
    
    if calendar is not None:
        try:
            gap = calendar.index(date) - calendar.index(nearest_month_end)
            if gap > max_stale_days:
                import warnings
                warnings.warn(
                    f"ETF features for {date} use cache from "
                    f"{nearest_month_end} ({gap} trading days ago).",
                    UserWarning
                )
        except ValueError:
            pass
    
    return cache[month_ends[idx]]

def build_feature_vector(etf_stl_feats: dict,
                         etf_vol_feats: dict,
                         stock_feats: dict) -> dict | None:
    """
    Construct full 292-dim feature vector:
      - etf_stl_feats : 240 features from get_etf_features
      - etf_vol_feats :  40 features from build_etf_volume_vol_features
      - stock_feats   :  12 features from build_stock_features
    """
    if not etf_stl_feats or etf_vol_feats is None or stock_feats is None:
        return None
    return {**etf_stl_feats, **etf_vol_feats, **stock_feats}