import datetime as dt

import pandas as pd
from util import get_data
from indicators import BB
import matplotlib.pyplot as plt
from marketsimcode import compute_portvals

class ManualStrategy():
    def __init__(self):
        pass

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "cqu41"  # replace tb34 with your Georgia Tech username.

    def testPolicy(self, symbol = "AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011,12,31), sv = 100000):

        #get stock data
        symbols = [symbol]
        date_range=pd.date_range(sd, ed)
        price_df = get_data(symbols, date_range)

        holdings = {symbol: 0}
        lookback = 14

        price_df = price_df[[symbol]]

        #build orders df
        orders= pd.DataFrame()
        orders['Date']=''
        orders['Symbol']=''
        orders['Order']=''
        orders['Shares']=''

        normed_price_df = price_df / price_df.iloc[0]
        # print(price_df)

        sma = normed_price_df.rolling(window=lookback).mean()

        # sma ratio from lecture
        sma_ratio = normed_price_df / sma #505 row 1 column

        #only use bb(percentage)
        upper_band, lower_band, bb = BB(lookback, normed_price_df)

        #get vix
        vix = get_data(['$VIX'], date_range, addSPY=True, colname='Adj Close')
        vix.fillna(method='ffill', inplace=True)
        vix.fillna(method='bfill', inplace=True)
        vix.drop(['SPY'], axis=1, inplace=True)
        # print(vix)
        # print("we're good here")

        for day in range(lookback, price_df.shape[0]):

            if (sma_ratio.ix[day, symbol] < 0.95) and (bb.ix[day, symbol] < 0) and vix.ix[day, '$VIX'] < 50 :
                if holdings[symbol] < 1000:
                    holdings[symbol] = holdings[symbol] + 1000
                    new_row = {'Date': price_df.index[day], 'Symbol': symbol, 'Order': 'BUY', 'Shares': 1000, 'Long entry': 1}
                    orders = orders.append(new_row, ignore_index=True)

            elif (sma_ratio.ix[day, symbol] > 1.05) and (bb.ix[day, symbol] > 1.0) and vix.ix[day, '$VIX'] > 20:
                if holdings[symbol] > -1000:
                    holdings[symbol] = holdings[symbol] - 1000
                    new_row = {'Date': price_df.index[day], 'Symbol': symbol, 'Order': 'SELL', 'Shares': 1000, 'Short entry': 1}
                    orders = orders.append(new_row, ignore_index=True)

            elif (sma_ratio.ix[day, symbol] >= 1) and (sma_ratio.ix[day-1, symbol] < 1) and (holdings[symbol] > 0) and vix.ix[day, '$VIX'] > 20:
                holdings[symbol] = 0
                new_row = {'Date': price_df.index[day], 'Symbol': symbol, 'Order': 'SELL', 'Shares': 1000}
                orders = orders.append(new_row, ignore_index=True)

            elif (sma_ratio.ix[day, symbol] <= 1) and (sma_ratio.ix[day-1, symbol] > 1) and (holdings[symbol] < 0) and vix.ix[day, '$VIX'] < 50:
                holdings[symbol] = 0
                new_row = {'Date': price_df.index[day], 'Symbol': symbol, 'Order': 'BUY', 'Shares': 1000}
                orders = orders.append(new_row, ignore_index=True)

        orders=orders.set_index(orders.columns[0])
        return orders

    def benchmark_orders(self, symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
        symbols = ['SPY']
        dates = pd.date_range(sd, ed)
        orders = get_data(symbols, dates=dates)

        orders['SPY'] = 0
        orders.rename(columns={'SPY': 'Symbol'}, inplace=True)
        orders['Order'] = 0
        orders['Shares'] = 0

        orders['Symbol'] = symbol
        orders['Order'] = 'BUY'
        orders.iloc[0, 2] = 1000

        # return orders dataframe
        return orders

