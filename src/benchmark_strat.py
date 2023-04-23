import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

%load_ext autoreload
%autoreload 2

data = pd.read_csv('snp_new.csv', index_col=0, parse_dates=True)
data = data[data.columns[data.columns.str.contains("Adj Close")]]
data.sort_values(by='Date', inplace=True)
data.dropna(axis=1, inplace=True)
data.drop("GOOG Adj Close", axis=1, inplace=True)
data.columns = data.columns.str.replace("Adj Close", "").str.strip()

# Functions Used

def normalising(data, formation_period, trading_period):
    '''
    This is simply normalising the data for the period given. As output you receive two dataframes of normalised data:
    1. formation period dataframe
    2. trading period dataframe
    '''
    normalised_data = (data+1).cumprod()
    formation_data = normalised_data.iloc[:formation_period]
    trading_data = normalised_data.iloc[formation_period:]
    return formation_data, trading_data

def sq_differences(normalized_v):
    """
    Creates a symmetrical matrix in which each value represents the squared difference between column i and index i.
    This function was created for the pairs_strategy() function
    """
    #creating a dataframe of n stocks * n stocks
    sqdiff_df = pd.DataFrame(columns=normalized_v.columns, index=normalized_v.columns)
    # filling up the empty dataframe with the squared differences.
    for c in sqdiff_df.columns:
        for i in sqdiff_df.index:
            ticker_c_normalisedP = normalized_v[c]
            ticker_i_normalisedP = normalized_v[i]
            sqdiff_ic = ((ticker_c_normalisedP - ticker_i_normalisedP)**2).sum()
            sqdiff_df.loc[i,c] = sqdiff_ic
    #Removing the zeros from the diagonal.
    sqdiff_df_nozero = sqdiff_df.query("@sqdiff_df>0")
    
    return sqdiff_df_nozero

def find_top_pairs(n_pairs, sqdiff_df_nozero):
    '''
    Provide the matrix of squared differences and the number of pairs you want. The output of this will be the top n_pairs of
    tickers in terms of the ones that have the lowest differences.
    '''
    pairs_list = []
    for i in range(1,n_pairs+1):
        minv = sqdiff_df_nozero.min().min()
        pair_tickers = sqdiff_df_nozero[sqdiff_df_nozero == minv].sum()[sqdiff_df_nozero[sqdiff_df_nozero == minv].sum() == minv].index
        pairs_list.append([pair_tickers[0], pair_tickers[1]])
        sqdiff_df_nozero = sqdiff_df_nozero.query("@sqdiff_df_nozero != @minv")
    
    return pairs_list

def signals(trading_data, current_stats, pairs_list):
    '''
    
    '''
    pairs = {}
    open_close = {}
    for n in np.linspace(1,current_stats.shape[0],current_stats.shape[0]):
        dif = trading_data[current_stats.loc[n, "pair"][0]] - trading_data[current_stats.loc[n, "pair"][1]]
        results_pair = pd.DataFrame(index = trading_data.index, columns = current_stats.loc[n, "pair"])
        length_pos = pd.DataFrame(index = trading_data.index, columns = ["open","close"])#NEW
        open_pos = []
        pos = "neutral" # this will change depending on whether we are long/short the first ticker in a pair
        
        for date in dif.index[:]:
            # if there is a signal to open a position, add 1 to the open_pos variable. 1 means there is an open position, nothing means
            # there is no open position
            if abs(dif.loc[date]) > current_stats.loc[n, "signal_open"] and 1 not in open_pos:
                open_pos.append(1)
                length_pos.loc[date,"open"] = 1 
            # when there is an active position and a signal to close it, remove the 1 from open_pos
                
            elif pos == "short" and dif.loc[date] < current_stats.loc[n, "signal_close"] and 1 in open_pos:
                open_pos.remove(1)
                length_pos.loc[date,"close"] = 1 
            elif pos == "long" and dif.loc[date] > current_stats.loc[n, "signal_close"] and 1 in open_pos:
                open_pos.remove(1)
                length_pos.loc[date,"close"] = 1 
                
            # adding a column that indicates whether there is an active position. len(open_pos) could only be 1 or 0.    
            len_open_pos = len(open_pos)
            results_pair.loc[date,:] = (-np.sign(dif.loc[date])*len_open_pos, len_open_pos*np.sign(dif.loc[date]))
            
            if -np.sign(dif.loc[date])*len_open_pos == -1:
                pos = "short" #when the above is negative, that means that we are short the first ticker in the pair
             
            elif -np.sign(dif.loc[date])*len_open_pos == 1:
                pos = "long" #when the above is positive, that means that we are long the first ticker in the pair
            else:
                pos = "neutral"
            
        pairs[n] = results_pair.shift(1) #shifting one assuming we enter the trade the next day after the signal
        length_pos["open"] = length_pos["open"].shift(1)
        open_close[n] = length_pos.replace(np.nan, 0) 
        
    return pairs, open_close

def rets_per_pair(data_rets, trading_data, signals, entries):
    """
    
    """
    rets_trading_data = data_rets.loc[trading_data.index, :]
    rets_pair_result = {}
    
    for n in np.linspace(1,len(signals),len(signals)):
        signals_per_pair = signals[n]
        entries_per_pair = entries[n]
        ticker1 = signals[n].columns[0]
        ticker2 = signals[n].columns[1]
        rets_trading_tickers = rets_trading_data[[ticker1,ticker2]]
        rets_strategy_pairs = (rets_trading_tickers*signals_per_pair)+1
        wealth_index = pd.DataFrame(1, index = trading_data.index, columns = [ticker1,ticker2])
    
        for date in rets_trading_data.index[:]:
            if entries_per_pair.loc[date, "open"] == 1:
                amount_per_ticker = (wealth_index.shift(1).loc[date].sum())/2
                wealth_index_day = amount_per_ticker * rets_strategy_pairs.loc[date]
                wealth_index.loc[date] = wealth_index_day
            else:
                if date == rets_trading_data.index[0]:
                    amount_per_ticker = [1, 1]
                    signals_per_pair.loc[date] = [0, 0]
                else:
                    amount_per_ticker = wealth_index.shift(1).loc[date]
                
                wealth_index.loc[date] = amount_per_ticker * rets_strategy_pairs.loc[date]
                    
                #print(amount_per_ticker, (rets_trading_tickers.loc[date]+1), signals_per_pair.loc[date])
                
        rets_pair_result[n] = wealth_index
        #print(n, wealth_index)
        
    return rets_pair_result
  
  def backtest(data, formation_period=252, trading_period=126, n_pairs = 10, spread_std_open = 2, spread_std_close = 0, comm = .0003):
    '''
    
    '''
    data_rets = data.pct_change()
    start_period = 0
    period = formation_period + trading_period
    n_periods = data.shape[0]
    
    start_df = data_rets.iloc[start_period:min(formation_period, n_periods)] #this is only used for the indexes of the below df
    index_1 = f"{start_df.index[0].to_period('d')} to {start_df.index[-1].to_period('d')}"
    indexes = [np.repeat(index_1,n_pairs), np.linspace(1, n_pairs, n_pairs)]
    columns = ["pair", "mean", "std", "signal_open", "signal_close"]
    historical_stats_df = pd.DataFrame(index = indexes, columns=columns)
    results_rets_per_pair = pd.DataFrame(index = data.iloc[formation_period:].index, 
                                         columns = np.linspace(1, n_pairs, n_pairs))
    commission_df = pd.DataFrame(0, index = data.iloc[formation_period:].index, 
                                         columns = np.linspace(1, n_pairs, n_pairs)) 
    
    while start_period+formation_period < n_periods:
        datarets_period = data_rets.iloc[start_period:min(start_period+period, n_periods)]
        # the below method is simply normalising the datarets_period df.
        formation_data, trading_data = normalising(datarets_period, formation_period=formation_period, 
                                                   trading_period=min(start_period+period, n_periods)-formation_period-start_period)

        def pairs_generation(formation_data=formation_data, n_pairs=n_pairs):
            '''
            Generates the pairs to be traded in the trading period
            '''
            # the below method is capturing in a matrix all the squared differences among every possible pair during
            # the formation period
            sqdiff_df = sq_differences(formation_data)
            pairs_list = find_top_pairs(n_pairs, sqdiff_df) # the first pair is the most cointegrated.
            return pairs_list
        pairs_list = pairs_generation()
        
        def historical_stats(df = historical_stats_df, formation_data = formation_data, spread_std_open = spread_std_open, spread_std_close = spread_std_close):
            '''
            Input: empty dataframe
            Output: historical stats including std and mean of squared deviations for every pair during the formation period.
            The output of this function will be used to create the dataframe with the signals (1, 0,-1)
            '''
            index_1 = f"{formation_data.index[0].to_period('d')} to {formation_data.index[-1].to_period('d')}"
            pair_n = 1
            
            for pairs in pairs_list:
                dif = (formation_data[pairs[0]] - formation_data[pairs[1]])
                mean = dif.mean()
                std = dif.std()
                signal_open = mean + std*spread_std_open
                signal_close = mean + std*spread_std_close
                df.loc[(index_1,pair_n),:] = (pairs, mean, std, signal_open, signal_close)
                pair_n += 1
                
            
            return df, index_1
        
        historical_stats_df,index_1 = historical_stats(df = historical_stats_df)
        historical_stats_current = historical_stats_df.loc[str(index_1)]
        signals_pairs, entry_exit = signals(trading_data=trading_data, current_stats=historical_stats_current, pairs_list=pairs_list)
        
        for n in np.linspace(1,len(entry_exit),len(entry_exit)):
            open_close = entry_exit[n]
            cost = (open_close * comm * 2).sum(axis=1)
            commission_df.loc[cost.index, n] = cost
        
        
        another(data=data_rets, trading_data=trading_data, signals=signals_pairs, entries=entry_exit) #using this code the below line works
        rets_per_pair_v = rets_per_pair(data_rets=data_rets, trading_data=trading_data, 
                                        signals=signals_pairs, entries=entry_exit)
        
        for n in np.linspace(1,len(rets_per_pair_v),len(rets_per_pair_v)):
            wealth_index_pair = rets_per_pair_v[n].sum(axis=1)
            ret_wealth_index = wealth_index_pair.pct_change()
            ret_wealth_index[0] = 0
            results_rets_per_pair.loc[wealth_index_pair.index, n] = ret_wealth_index
        # when this formation period and trading period end, let's compute the following
        start_period = start_period + trading_period
        #end = end + formation_period + trading_period
    return pairs_list, historical_stats_df, signals_pairs, entry_exit, rets_per_pair_v, results_rets_per_pair, commission_df
  
################################################################################################################################################
################################################################################################################################################
  
#Backtesting
  
pairs, stats, signals_v, entries, rets_temp, final_results, cost_per_pair = backtest(data)
  
(final_results+1).prod()
final_results_after_cost = final_results - cost_per_pair
(final_results_after_cost+1).prod()
  
(final_results_after_cost+1).cumprod().plot(figsize=(14,8))
plt.ylabel("Cumulative Return")
plt.title("Cumulative Returns (Incl. Trading Costs)")

(final_results_df+1).cumprod().plot(figsize=(14,8))
plt.ylabel("Cumulative Return")
plt.title("Cumulative Returns (Excl. Trading Costs)")

value_portfolio = pd.concat([(final_results_after_cost+1).prod(), (final_results_df+1).prod()], axis=1)
value_portfolio.columns = ["Incl. Trading Costs","Excl. Trading Costs"]
value_portfolio

# Alternative Dataset

df_prices2 = pd.read_csv(r"prices_long.csv", index_col="Date", parse_dates=True)
pairs2, stats2, signals_v2, entries2, rets_temp2, final_results2 = backtest(df_prices2.dropna(axis=1))

alt_value_portfolio = pd.concat([(final_results2+1).prod(), (final_results_df+1).prod()], axis=1)
alt_value_portfolio.columns = ["Alternative Dataset","Excl. Trading Costs"]
alt_value_portfolio

(final_results2+1).cumprod().plot(figsize=(14,8))
plt.ylabel("Cumulative Return")
plt.title("Cumulative Returns - Alternative dataset")

################################################################################################################################################
################################################################################################################################################
  
#Computing Summary Stats

def drawdown(return_series: pd.Series):
    """Takes a time series of asset returns.
       returns a DataFrame with columns for
       the wealth index, 
       the previous peaks, and 
       the percentage drawdown
    """
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({"Wealth": wealth_index, 
                         "Previous Peak": previous_peaks, 
                         "Drawdown": drawdowns})

def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3

def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4

def annualize_rets(r, periods_per_year):
    """
    Annualizes a set of returns
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1

def annualize_vol(r, periods_per_year):
    """
    Annualizes the vol of a set of returns
    """
    return r.std()*(periods_per_year**0.5)

def sharpe_ratio(r, riskfree_rate, periods_per_year):
    """
    Computes the annualized sharpe ratio of a set of returns
    """
    # convert the annual riskfree rate to per period
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol

from scipy.stats import norm
def var_gaussian(r, level=5, modified=False):
    """
    Returns the Parametric Gauusian VaR of a Series or DataFrame
    If "modified" is True, then the modified VaR is returned,
    using the Cornish-Fisher modification
    """
    # compute the Z score assuming it was Gaussian
    z = norm.ppf(level/100)
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
                (z**2 - 1)*s/6 +
                (z**3 -3*z)*(k-3)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )
    return -(r.mean() + z*r.std(ddof=0))

def cvar_historic(r, level=5):
    """
    Computes the Conditional VaR of Series or DataFrame
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")

def var_historic(r, level=5):
    """
    Returns the historic Value at Risk at a specified level
    i.e. returns the number such that "level" percent of the returns
    fall below that number, and the (100-level) percent are above
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


def summary_stats(r, periods_per_year,riskfree_rate=0.03):
    """
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
    """
    ann_r = r.aggregate(annualize_rets, periods_per_year=periods_per_year)
    ann_vol = r.aggregate(annualize_vol, periods_per_year=periods_per_year)
    ann_sr = r.aggregate(sharpe_ratio, riskfree_rate=riskfree_rate, periods_per_year=periods_per_year)
    dd = r.aggregate(lambda r: drawdown(r).Drawdown.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaussian, modified=True)
    hist_cvar5 = r.aggregate(cvar_historic)
    return pd.DataFrame({
       "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Historic CVaR (5%)": hist_cvar5,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd
    })
  
  (final_results+1).cumprod().plot(figsize=(12,6))
  (final_results2+1).cumprod().plot(figsize=(12,6))
  
  summary_stats(final_results_df, 252)
  summary_stats(final_results2, 252)
