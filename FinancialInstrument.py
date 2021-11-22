#!/usr/bin/env python
# coding: utf-8

# FinancialInstrument Class


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from itertools import product
plt.style.use("seaborn")



class FinancialInstrument():
    
    def __init__(self, ticker, start=None, end=None):
        self._ticker = ticker
        self.start = start
        self.end = end
        self.get_data()
        self.log_returns()
        
    def __repr__(self):
        return "FinancialInstrument(ticker = {}, start = {}, end = {})".format(self._ticker, 
                                                                              self.start,
                                                                              self.end)
        
    def get_data(self):
        raw_data = yf.download(self._ticker, self.start, self.end).Close.to_frame()
        raw_data.rename({"Close": "price"}, axis='columns', inplace=True)
        self.data = raw_data
    
    def log_returns(self):
        self.data['log_returns'] = np.log(self.data.price/self.data.price.shift(1))
        
    def plot_prices(self):
        self.data.price.plot(figsize=(12,8))
        plt.title("Price Chart: {}".format(self._ticker), fontsize = 15)
    
    def plot_returns(self, kind="ts"):
        if kind == "ts":
            self.data.log_returns.plot(figsize=(12, 8))
            plt.title("Returns: {}".format(self._ticker), fontsize = 15)
        elif kind == "hist":
            self.data.log_returns.hist(figsize=(12, 8), bins = int(np.sqrt(len(self.data))))
            plt.title("Frequency of Returns: {}".format(self._ticker), fontsize = 15)
        
    def set_ticker(self, ticker = None):
        if ticker is not None:
            self._ticker = ticker
            self.get_data()
            self.log_returns()
            
    def set_start(self, start = None):
        if start is not None:
            self.start = start
            self.get_data()
            self.log_returns()
            
    def set_end(self, end = None):
        if end is not None:
            self.end = end
            self.get_data()
            self.log_returns()




class RiskReturn(FinancialInstrument): # Child
    
    def __init__(self, ticker, start, end, freq = None):
        self.freq = freq
        super().__init__(ticker, start, end)
    
    
    def __repr__(self): 
        return "RiskReturn(ticker = {}, start = {}, end = {})".format(self._ticker, 
                                                                          self.start, self.end)
    def mean_return(self):
        if self.freq is None:
            return self.data.log_returns.mean()
        else:
            resampled_price = self.data.price.resample(self.freq).last()
            resampled_returns = np.log(resampled_price / resampled_price.shift(1))
            return resampled_returns.mean()
    
    def std_returns(self):
        if self.freq is None:
            return self.data.log_returns.std()
        else:
            resampled_price = self.data.price.resample(self.freq).last()
            resampled_returns = np.log(resampled_price / resampled_price.shift(1))
            return resampled_returns.std()
        
    def annualized_perf(self):
        mean_return = round(self.data.log_returns.mean() * 252, 3)
        risk = round(self.data.log_returns.std() * np.sqrt(252), 3)
        print("Return: {} | Risk: {}".format(mean_return, risk))
        #print("The sharpe ratio is {}".format(mean))        
        
        
        
        
class EMABackTester(FinancialInstrument):
    def __init__(self, ticker, EMA_S, EMA_L, start=None, end=None):
        self.EMA_S = EMA_S
        self.EMA_L = EMA_L
        self.results = None
        self._ticker = ticker
        self.start = start
        self.end = end
        self.get_data()
        self.log_returns()
        
        
    def __repr__(self): 
        return "EMABackTester(ticker = {}, start = {}, end = {})".format(self._ticker, self.start, self.end)
    
    def get_data(self): # it overwrites the old function
        super().get_data()
        self.prepare_data()
    
    def prepare_data(self):
        #data = self.data.copy()
        self.data['EMA_S'] = self.data['price'].ewm(span = self.EMA_S, min_periods= self.EMA_S).mean()
        self.data['EMA_L'] = self.data['price'].ewm(span = self.EMA_L, min_periods= self.EMA_L).mean()
        #self.data = data
    
    def log_returns(self):
        super().log_returns()
        self.data.rename({'log_returns': 'returns'}, axis = 'columns', inplace=True)
        self.data = self.data[['price', 'returns', 'EMA_S', 'EMA_L']]
        
    def set_parameters(self, EMA_S = None, EMA_L = None):
        '''reset short and long
        '''
        if EMA_S is not None: self.EMA_S = EMA_S 
        if EMA_L is not None: self.EMA_L = EMA_L
        if EMA_S is not None or EMA_L is not None:
            self.prepare_data()
 
    def test_strategy(self):
        data = self.data.copy().dropna()
        data['position'] = np.where(data['EMA_S'] > data['EMA_L'], 1, -1)
        #data['position'].shift(1)
        #return data
        data['strategy'] = data['position'].shift(1) * data['returns']
        data.dropna(inplace=True)
        data['creturns'] = data['returns'].cumsum().apply(np.exp)
        data['cstrategy'] = data['strategy'].cumsum().apply(np.exp)
        self.results = data
        
        perf = data['cstrategy'].iloc[-1]
        outperf = perf - data['creturns'].iloc[-1]
        #print("The performance of the strategy EMA_S: {} EMA_L: {} is {}, \n outperformed than simple buy and hold by {}".format(self.EMA_S, self.EMA_L, round(perf, 6), round(outperf, 6)))
        return round(perf, 6), round(outperf, 6)
    
    def plot_results(self):
        if self.results is None:
            print("Please run test_strategy() first")
        else:
            title = "{} | EMA_S = {} | EMA_L = {}".format(self._ticker, self.EMA_S, self.EMA_L)
            self.results[['creturns', 'cstrategy']].plot(title=title, figsize=(12,8))
    
    def optimize_parameters(self, EMA_S_range, EMA_L_range):
        ''' Finds the optimal strategy (global maximum) given the EMA parameter ranges.

        Parameters
        ----------
        EMA_S_range, EMA_L_range: tuple
            tuples of the form (start, end, step size)
        '''
        combinations = list(product(range(*EMA_S_range), range(*EMA_L_range)))
        
        # test all combinations
        results = []
        for comb in combinations:
            self.set_parameters(comb[0], comb[1])
            results.append(self.test_strategy()[0])
        
        best_perf = np.max(results) # best performance
        opt = combinations[np.argmax(results)] # optimal parameters
        
        # run/set the optimal strategy
        self.set_parameters(opt[0], opt[1])
        self.test_strategy()
                   
        # create a df with many results
        many_results =  pd.DataFrame(data = combinations, columns = ["EMA_S", "EMA_L"])
        many_results["performance"] = results
        self.results_overview = many_results
                            
        return opt, best_perf
    
    
class ConBacktester():
    ''' Class for the vectorized backtesting of simple contrarian trading strategies.
    '''    
    
    def __init__(self, symbol, start, end, tc):
        '''
        Parameters
        ----------
        symbol: str
            ticker symbol (instrument) to be backtested
        start: str
            start date for data import
        end: str
            end date for data import
        tc: float
            proportional transaction/trading costs per trade
        '''
        self.symbol = symbol
        self.start = start
        self.end = end
        self.tc = tc
        self.results = None
        self.get_data()
        
    def __repr__(self):
        return "ConBacktester(symbol = {}, start = {}, end = {})".format(self.symbol, self.start, self.end)
        
    def get_data(self):
        ''' Imports the data from intraday_pairs.csv (source can be changed).
        '''
        raw = pd.read_csv("intraday_pairs.csv", parse_dates = ["time"], index_col = "time")
        raw = raw[self.symbol].to_frame().dropna()
        raw = raw.loc[self.start:self.end].copy()
        raw.rename(columns={self.symbol: "price"}, inplace=True)
        raw["returns"] = np.log(raw / raw.shift(1))
        self.data = raw
        
    def test_strategy(self, window = 1):
        ''' Backtests the simple contrarian trading strategy.
        
        Parameters
        ----------
        window: int
            time window (number of bars) to be considered for the strategy.
        '''
        self.window = window
        data = self.data.copy().dropna()
        data["position"] = -np.sign(data["returns"].rolling(self.window).mean())
        data["strategy"] = data["position"].shift(1) * data["returns"]
        data.dropna(inplace=True)
        
        # determine the number of trades in each bar
        data["trades"] = data.position.diff().fillna(0).abs()
        
        # subtract transaction/trading costs from pre-cost return
        data.strategy = data.strategy - data.trades * self.tc
        
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data
        
        perf = data["cstrategy"].iloc[-1] # absolute performance of the strategy
        outperf = perf - data["creturns"].iloc[-1] # out-/underperformance of strategy
        
        return round(perf, 6), round(outperf, 6)
    
    def plot_results(self):
        ''' Plots the performance of the trading strategy and compares to "buy and hold".
        '''
        if self.results is None:
            print("Run test_strategy() first.")
        else:
            title = "{} | Window = {} | TC = {}".format(self.symbol, self.window, self.tc)
            self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(12, 8))
            
    def optimize_parameter(self, window_range):
        ''' Finds the optimal strategy (global maximum) given the window parameter range.

        Parameters
        ----------
        window_range: tuple
            tuples of the form (start, end, step size)
        '''
        
        windows = range(*window_range)
            
        results = []
        for window in windows:
            results.append(self.test_strategy(window)[0])
        
        best_perf = np.max(results) # best performance
        opt = windows[np.argmax(results)] # optimal parameter
        
        # run/set the optimal strategy
        self.test_strategy(opt)
        
        # create a df with many results
        many_results =  pd.DataFrame(data = {"window": windows, "performance": results})
        self.results_overview = many_results
        
        return opt, best_perf
                               
