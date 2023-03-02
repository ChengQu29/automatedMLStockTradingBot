""""""
import math

"""MC1-P2: Optimize a portfolio.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
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
  		  	   		  		 			  		 			     			  	 
Student Name: Chengwen Qu  		  	   		  		 			  		 			     			  	 
GT User ID: cqu41 		  	   		  		 			  		 			     			  	 
GT ID: 903756933 		  	   		  		 			  		 			     			  	 
"""


import datetime as dt

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from util import get_data, plot_data
import scipy.optimize as spo


# This is the function that will be tested by the autograder
def optimize_portfolio(
    sd=dt.datetime(2008, 1, 1, 0, 0),
    ed=dt.datetime(2009, 1, 1, 0, 0),
    syms=["GOOG", "AAPL", "GLD", "XOM"],
    gen_plot=False,
):
    """
    This function should find the optimal allocations for a given set of stocks. You should optimize for maximum Sharpe
    Ratio. The function should accept as input a list of symbols as well as start and end dates and return a list of
    floats (as a one-dimensional numpy array) that represents the allocations to each of the equities. You can take
    advantage of routines developed in the optional assess portfolio project to compute daily portfolio value and
    statistics.
    """

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    #print(type(prices_all))
    prices_all.fillna(method='ffill')
    prices_all.fillna(method='bfill')
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later
    #print(prices)


    num_of_assets = len(syms)
    initial_guess_allocs = np.asarray([1.0/num_of_assets] * num_of_assets)
    # the minimizer takes a function to minimize and returns argument for which it minimizes
    optimized_allocs = spo.minimize(
              compute_sharpe_ratio,
              initial_guess_allocs,
              args=(prices), # extra args
              method='SLSQP',
              bounds=[(0,1)]*len(syms),
              constraints=({'type': 'eq', 'fun': lambda inputs: 1.0 - np.sum(inputs)}),
              options={'disp':True})

    portfolio_stats = assess_portfolio(optimized_allocs['x'], prices) # returns a tuple

    cr = portfolio_stats[0]
    adr = portfolio_stats[1]
    sddr = portfolio_stats[2]
    sr = portfolio_stats[3]
    port_val = portfolio_stats[4]
    #print(port_val)

    if gen_plot:
        normed_SPY = prices_SPY / prices_SPY.iloc[0]
        df_temp = pd.concat(
            [port_val, normed_SPY], keys=["Portfolio", "SPY"], axis=1
        )
        plot_data(df_temp, title="Daily Portfolio Value and SPY", xlabel="Date",
                  ylabel="Normalized price")

        plot_and_save(df_temp, "Daily Portfolio Value and SPY", "Date", "Normalized price")

    return optimized_allocs['x'], cr, adr, sddr, sr

def plot_and_save(df, title, xlabel, ylabel):
    yy = df.plot(title=title, grid=True)
    yy.set_xlabel(xlabel)
    yy.set_ylabel(ylabel)
    plt.savefig("Figure1.png")


# this function takes allocations and starting value?
# this function returns sharpe ratio * -1
def compute_sharpe_ratio(allocs, prices):

    portfolio_stats = assess_portfolio(allocs, prices)

    # return sharpe ratio * -1
    return portfolio_stats[3] * -1

def assess_portfolio(allocs, prices):
    """
    Estimate a set of test points given the model we built.
    :return: A tuple containing the cumulative return, average daily returns,
        standard deviation of daily returns, Sharpe ratio and end value
    :rtype: tuple
    """

    # Get daily portfolio value
    #port_val = prices_SPY  # add code here to compute daily portfolio values

    #opportunity for refactoring:
    sf = 252
    rfr = 0.0

    normed = prices/prices.iloc[0]
    alloced = normed * allocs
    #pos_vals = alloced * 10000000 # sv:10000000
    #port_val = pos_vals.sum(axis=1) # sum each row
    port_val = alloced.sum(axis=1)
    #daily return
    daily_ret = port_val/port_val.shift(1) - 1
    #print(daily_ret)

    #compute cumulative return
    cr = (port_val.iloc[-1]/port_val.iloc[0]) - 1
    #compute average daily return
    adr = daily_ret.mean()
    #compute std of daily return (volatility)
    sddr = daily_ret.std()

    #compute sharpe ratio
    sr = math.sqrt(sf) * (adr-rfr) / sddr


    return cr, adr, sddr, sr, port_val



def test_code():
    """
    This function WILL NOT be called by the auto grader.
    """

    start_date = dt.datetime(2008, 6, 1)
    end_date = dt.datetime(2009, 6, 1)
    symbols = ["IBM", "X", "GLD", "JPM"]

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(
        sd=start_date, ed=end_date, syms=symbols, gen_plot=True
    )

    # Print statistics
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    print(f"Symbols: {symbols}")
    print(f"Allocations:{allocations}")
    print(f"Sharpe Ratio: {sr}")
    print(f"Volatility (stdev of daily returns): {sddr}")
    print(f"Average Daily Return: {adr}")
    print(f"Cumulative Return: {cr}")


if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader
    # Do not assume that it will be called
    test_code()
