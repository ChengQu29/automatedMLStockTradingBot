""""""  		  	   		  		 			  		 			     			  	 
"""  		  	   		  		 			  		 			     			  	 
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  		 			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		  		 			  		 			     			  	 
All Rights Reserved  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Template code for CS 4646/7646  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  		 			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		  		 			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		  		 			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		  		 			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		  		 			  		 			     			  	 
or edited.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		  		 			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		  		 			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  		 			  		 			     			  	 
GT honor code violation.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
-----do not edit anything above this line---  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Student Name: Tucker Balch (replace with your name)  		  	   		  		 			  		 			     			  	 
GT User ID: cqu41 (replace with your User ID)  		  	   		  		 			  		 			     			  	 
GT ID: 903756933 (replace with your GT ID)  		  	   		  		 			  		 			     			  	 
"""  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
import datetime as dt  		  	   		  		 			  		 			     			  	 
import random  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
import pandas as pd  		  	   		  		 			  		 			     			  	 
import util as ut
from util import get_data
from  indicators import BB
import RTLearner as rtl
import BagLearner as bgl
import numpy as np
  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
class StrategyLearner(object):  		  	   		  		 			  		 			     			  	 
    """  		  	   		  		 			  		 			     			  	 
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  		 			  		 			     			  	 
        If verbose = False your code should not generate ANY output.  		  	   		  		 			  		 			     			  	 
    :type verbose: bool  		  	   		  		 			  		 			     			  	 
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		  		 			  		 			     			  	 
    :type impact: float  		  	   		  		 			  		 			     			  	 
    :param commission: The commission amount charged, defaults to 0.0  		  	   		  		 			  		 			     			  	 
    :type commission: float  		  	   		  		 			  		 			     			  	 
    """  		  	   		  		 			  		 			     			  	 
    # constructor  		  	   		  		 			  		 			     			  	 
    def __init__(self, verbose=False, impact=0.0, commission=0.0):  		  	   		  		 			  		 			     			  	 
        """  		  	   		  		 			  		 			     			  	 
        Constructor method  		  	   		  		 			  		 			     			  	 
        """  		  	   		  		 			  		 			     			  	 
        self.verbose = verbose  		  	   		  		 			  		 			     			  	 
        self.impact = impact  		  	   		  		 			  		 			     			  	 
        self.commission = commission
        self.learner = bgl.BagLearner(learner = rtl.RTLearner, kwargs={'leaf_size': 5}, bags = 20, boost = False, verbose = False)

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "cqu41"  # replace tb34 with your Georgia Tech username.

    def prepare_data(self, symbol, sd, ed, sv):

        # get stock data
        symbols = [symbol]
        date_range = pd.date_range(sd, ed)
        price_df = get_data(symbols, date_range)

        lookback = 14

        price_df = price_df[[symbol]]

        normed_price_df = price_df / price_df.iloc[0]
        # print(price_df)

        sma = normed_price_df.copy().rolling(window=lookback).mean()

        # sma ratio from lecture
        sma_ratio = normed_price_df / sma  # 505 row 1 column
        sma_ratio.fillna(method='ffill', inplace=True)
        sma_ratio.fillna(method='bfill', inplace=True)
        sma_ratio.columns=['p/sma']


        # only use bb(percentage)
        upper_band, lower_band, bb = BB(lookback, normed_price_df)
        bb.fillna(method='ffill', inplace=True)
        bb.fillna(method='bfill', inplace=True)
        bb.columns=['bb']

        # get vix
        vix = get_data(['$VIX'], date_range, addSPY=True, colname='Adj Close')
        vix.fillna(method='ffill', inplace=True)
        vix.fillna(method='bfill', inplace=True)
        vix.drop(['SPY'], axis=1, inplace=True)

        combined_feature_df = pd.concat([sma_ratio, bb, vix], axis=1)



        # build a date_index of the same shape as ...
        date_index = combined_feature_df.index

        # print(combined_feature_df)
        x_data = combined_feature_df.values[:, 0:]  # x_train is now an np array


        future_price = price_df.shift(-lookback)
        current_price = price_df
        y_data = (future_price / current_price) - 1
        positive_mask = y_data > 0
        negative_mask = y_data < 0
        y_data[positive_mask] = price_df.shift(-lookback) / (price_df * (1.0 + self.impact)) - 1.0
        y_data[negative_mask] = price_df.shift(-lookback) / (price_df * (1.0 - self.impact)) - 1.0

        y_data = y_data.values

        return x_data, y_data, date_index, normed_price_df


    # this method should create a QLearner, and train it for trading  		  	   		  		 			  		 			     			  	 
    def add_evidence(  		  	   		  		 			  		 			     			  	 
        self,  		  	   		  		 			  		 			     			  	 
        symbol="IBM",  		  	   		  		 			  		 			     			  	 
        sd=dt.datetime(2008, 1, 1),  		  	   		  		 			  		 			     			  	 
        ed=dt.datetime(2009, 12, 31),
        sv=10000,  		  	   		  		 			  		 			     			  	 
    ):  		  	   		  		 			  		 			     			  	 
        """  		  	   		  		 			  		 			     			  	 
        Trains your strategy learner over a given time frame.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
        :param symbol: The stock symbol to train on  		  	   		  		 			  		 			     			  	 
        :type symbol: str  		  	   		  		 			  		 			     			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  		 			  		 			     			  	 
        :type sd: datetime  		  	   		  		 			  		 			     			  	 
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  		 			  		 			     			  	 
        :type ed: datetime  		  	   		  		 			  		 			     			  	 
        :param sv: The starting value of the portfolio  		  	   		  		 			  		 			     			  	 
        :type sv: int  		  	   		  		 			  		 			     			  	 
        """  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
        x_train, y_train, date_index, not_used_price_df = self.prepare_data(symbol, sd, ed, sv)

        y_train=y_train.flatten() #(303,)
        # print(y_train.shape)
        # print(y_train)

        #classification
        buy_threshold=0.008
        sell_threshold=-0.008

        y_train[y_train>buy_threshold]=1
        y_train[y_train<sell_threshold]=-1
        y_train[(y_train >= sell_threshold) & (y_train <= buy_threshold)] = 0
        # print(y_train)
        # print(y_train.shape)

        #call learner

        d_tree=self.learner.add_evidence(x_train,y_train)
        target_y=self.learner.query(x_train)
        # print(target_y)

  		  	   		  		 			  		 			     			  	 
    # this method should use the existing policy and test it against new data  		  	   		  		 			  		 			     			  	 
    def testPolicy(  		  	   		  		 			  		 			     			  	 
        self,  		  	   		  		 			  		 			     			  	 
        symbol="IBM",  		  	   		  		 			  		 			     			  	 
        sd=dt.datetime(2010, 1, 1),
        ed=dt.datetime(2011, 12, 31),
        sv=10000,  		  	   		  		 			  		 			     			  	 
    ):  		  	   		  		 			  		 			     			  	 
        """  		  	   		  		 			  		 			     			  	 
        Tests your learner using data outside of the training data  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
        :param symbol: The stock symbol that you trained on on  		  	   		  		 			  		 			     			  	 
        :type symbol: str  		  	   		  		 			  		 			     			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  		 			  		 			     			  	 
        :type sd: datetime  		  	   		  		 			  		 			     			  	 
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  		 			  		 			     			  	 
        :type ed: datetime  		  	   		  		 			  		 			     			  	 
        :param sv: The starting value of the portfolio  		  	   		  		 			  		 			     			  	 
        :type sv: int  		  	   		  		 			  		 			     			  	 
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		  		 			  		 			     			  	 
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		  		 			  		 			     			  	 
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		  		 			  		 			     			  	 
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		  		 			  		 			     			  	 
        :rtype: pandas.DataFrame  		  	   		  		 			  		 			     			  	 
        """  		  	   		  		 			  		 			     			  	 

        # get data
        x_test, not_used, date_index, normed_price_df = self.prepare_data(symbol, sd, ed, sv)

        y_test = self.learner.query(x_test)
        y_test = np.nan_to_num(y_test)

        # next, generate orders
        trades = normed_price_df.copy()
        trades.iloc[:, :] = 0
        position = 0

        for i in range(len(trades)):

            position += trades.iloc[i - 1, trades.columns.get_loc(symbol)]

            if y_test[i] <0 :
                if position == 0:
                    trades.iloc[i, :] = -1000
                elif position == 1000:
                    trades.iloc[i, :] = -2000
                else:
                    trades.iloc[i, :] = 0
            elif y_test[i] >0:
                if position == 0:
                    trades.iloc[i, :] = 1000
                elif position == 1000:
                    trades.iloc[i, :] = 0
                else:
                    trades.iloc[i, :] = 2000
            else:
                trades.iloc[i, :] = 0
        # print(trades)
        # print(type(trades))
        return trades

    def generate_marketsim_orders(self, symbol, trades):
        orders=pd.DataFrame(index=trades.index, columns=['Symbol', 'Order', 'Shares'])

        for index, row in trades.iterrows():
            shares=row['JPM']
            if shares>0:
                orders.loc[index]=[symbol, 'BUY', abs(shares)]
            if shares<0:
                orders.loc[index] = [symbol, 'SELL', abs(shares)]

        orders=orders.dropna(subset=['Order'])
        # print(orders)
        return orders

if __name__ == "__main__":
    test_learner = StrategyLearner()
    test_learner.add_evidence(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=10000)
    trades = test_learner.testPolicy(symbol="JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=10000)
    test_learner.generate_marketsim_orders('JPM', trades)
    print("One does not simply think up a strategy")
