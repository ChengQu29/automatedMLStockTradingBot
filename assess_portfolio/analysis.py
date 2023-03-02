"""Analyze a portfolio.

Copyright 2017, Georgia Tech Research Corporation
Atlanta, Georgia 30332-0415
All Rights Reserved
"""

import datetime as dt

import numpy as np

import pandas as pd
from numpy import sqrt

from util import get_data, plot_data


# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def assess_portfolio(
    sd=dt.datetime(2008, 1, 1),
    ed=dt.datetime(2009, 1, 1),
    syms=["GOOG", "AAPL", "GLD", "XOM"],
    allocs=[0.1, 0.2, 0.3, 0.4],
    sv=1000000,
    rfr=0.0,
    sf=252.0,
    gen_plot=True,
):
    """  		  	   		  		 			  		 			     			  	 
    Estimate a set of test points given the model we built.
    :return: A tuple containing the cumulative return, average daily returns,  		  	   		  		 			  		 			     			  	 
        standard deviation of daily returns, Sharpe ratio and end value  		  	   		  		 			  		 			     			  	 
    :rtype: tuple  		  	   		  		 			  		 			     			  	 
    """

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices_all.fillna(method='ffill')
    prices_all.fillna(method='bfill')
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # Get daily portfolio value
    #port_val = prices_SPY  # add code here to compute daily portfolio values

    print("allocations: ", allocs) #if not specify allocs when calling the function, the default value will be [0.1, 0.2, 0.3, 0.4] as stated in the function signature

    normed = prices/prices.iloc[0]
    alloced = normed * allocs
    pos_vals = alloced * sv
    port_val = pos_vals.sum(axis=1) # sum each row
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
    sr = sqrt(sf) * (adr-rfr) / sddr

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        normed_SPY = prices_SPY / prices_SPY.iloc[0]
        df_temp = pd.concat(
            [alloced.sum(axis=1), normed_SPY], keys=["Portfolio", "SPY"], axis=1
        )
        plot_data(df_temp, title="Daily Portfolio Value and SPY", xlabel="Date",
                  ylabel="Normalized price")
        #plot_data(df_temp, title="Daily Portfolio Value and SPY")

    # Add code here to properly compute end value
    ev = port_val.iloc[-1]

    return cr, adr, sddr, sr, ev


def test_code():
    """
    Performs a test of your code and prints the results
    """
    # This code WILL NOT be tested by the auto grader
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!
    start_date = dt.datetime(2009, 1, 1)
    end_date = dt.datetime(2010, 1, 1)
    symbols = ["GOOG", "AAPL", "GLD", "XOM"]
    allocations = [0.2, 0.3, 0.4, 0.1]
    start_val = 1000000
    risk_free_rate = 0.0
    sample_freq = 252

    # Assess the portfolio
    cr, adr, sddr, sr, ev = assess_portfolio(
        sd=start_date,
        ed=end_date,
        syms=symbols,
        allocs=allocations,
        sv=start_val,
        gen_plot=True,
    )

    # Print statistics
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    print(f"Symbols: {symbols}")
    print(f"Allocations: {allocations}")
    print(f"Sharpe Ratio: {sr}")
    print(f"Volatility (stdev of daily returns): {sddr}")
    print(f"Average Daily Return: {adr}")
    print(f"Cumulative Return: {cr}")


if __name__ == "__main__":
    test_code()
