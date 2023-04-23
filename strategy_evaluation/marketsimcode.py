""""""
"""MC2-P1: Market simulator.  		  	   		  		 			  		 			     			  	 

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

Student Name: Chengwen Qu (replace with your name)  		  	   		  		 			  		 			     			  	 
GT User ID: cqu41 (replace with your User ID)  		  	   		  		 			  		 			     			  	 
GT ID: 903756933  (replace with your GT ID)  		  	   		  		 			  		 			     			  	 
"""

import datetime as dt

import numpy as np

import pandas as pd
from util import get_data, plot_data


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "cqu41"  # replace tb34 with your Georgia Tech username.


def compute_portvals(
        orders_df,
        start_val=1000000,
        commission=9.95,
        impact=0.005,
):
    """
    Computes the portfolio values.

    :param orders_file: Path of the order file or the file object
    :type orders_file: str or file object
    :param start_val: The starting value of the portfolio
    :type start_val: int
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)
    :type commission: float
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction
    :type impact: float
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.
    :rtype: pandas.DataFrame
    """
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input
    # TODO: Your code here

    # step 1 Prices dataframe:
    datetime_index = orders_df.index
    # print(type(datetime_index))
    # print(orders_df['Symbol'])

    # collect symbols from order
    # print(orders_df['Symbol'].unique())
    symbols = orders_df['Symbol'].unique()

    start_date = orders_df.index.values[0]
    end_date = orders_df.index.values[-1]
    # df_prices = build_prices_df(orders_df, symbols, datetime_index)
    df_prices = get_data(symbols, pd.date_range(start_date, end_date))
    # print(df_prices)
    # print(orders_df)

    # step 2 Trades dataframe:
    df_trades = build_trades_df(df_prices, orders_df, commission, impact, start_val)
    # print(df_trades)

    # step 3 Holdings dataframe:
    df_holdings = build_holdings_df(df_trades)
    # print(df_holdings)

    # step 4 Values dataframe:
    return build_values(df_holdings, df_prices)


def build_values(df_holdings, df_prices):
    df_prices['CASH'] = 1.0
    value_df = df_prices * df_holdings

    port_val = value_df.sum(axis=1)

    return port_val


# build holdings dataframe
def build_holdings_df(df_trades):
    for i in range(1, df_trades.shape[0]):
        df_trades.iloc[i] += df_trades.iloc[i - 1]
    return df_trades


# build trades dataframe
def build_trades_df(df_prices, orders_df, commission, impact, start_val):
    # take the prices dataframe and fill every row with zeros except the index row
    new_df = pd.DataFrame(np.zeros(df_prices.shape), columns=df_prices.columns, index=df_prices.index)
    new_df = new_df.assign(zeros=pd.Series(0.0, index=df_prices.index))
    new_df_with_cash = new_df.rename(columns={'zeros': 'CASH'})
    # print(new_df_with_cash)

    new_df_with_cash['CASH'][0] = start_val
    # print(new_df_with_cash)
    # put the orders number in
    for index, row in orders_df.iterrows():
        # print(index)
        # print(type(row))

        symbol = row['Symbol']

        if row['Order'] == 'BUY':
            new_df_with_cash.at[index, symbol] += row['Shares']
            new_df_with_cash.at[index, 'CASH'] -= (row['Shares'] * df_prices.at[index, symbol] * (1 + impact))
            new_df_with_cash.at[index, 'CASH'] -= commission

        else:
            new_df_with_cash.at[index, symbol] -= row['Shares']
            new_df_with_cash.at[index, 'CASH'] += (row['Shares'] * df_prices.at[index, symbol] * (1 - impact))
            new_df_with_cash.at[index, 'CASH'] -= commission

    return new_df_with_cash


# find the prices for symbols in orders and put it in a prices dataframe
def build_prices_df(orders_df, symbols, datetime_index):
    df_prices = pd.DataFrame(index=datetime_index)

    for sym in symbols:
        # create a dataframe with the DatetimeIndex as the index for that symbol
        df_temp = pd.read_csv(f"../data/{sym}.csv", index_col='Date', parse_dates=True,
                              usecols=['Date', 'Adj Close'],
                              na_values=['nan'])
        index_intersection = df_temp.index.intersection(orders_df.index)
        price_for_symbol = df_temp.loc[index_intersection]
        price_for_symbol_renamed = price_for_symbol.rename(columns={'Adj Close': sym})
        price_for_symbol_reverse = price_for_symbol_renamed[::-1]
        df_prices = df_prices.join(price_for_symbol_reverse)

    return df_prices.drop_duplicates()


def test_code():
    """
    Helper function to test code
    """
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders-01.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file=of, start_val=sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2008, 6, 1)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [
        0.2,
        0.01,
        0.02,
        1.5,
    ]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [
        0.2,
        0.01,
        0.02,
        1.5,
    ]

    # Compare portfolio against $SPX
    print(f"Date Range: {start_date} to {end_date}")
    print()
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")
    print()
    print(f"Cumulative Return of Fund: {cum_ret}")
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")
    print()
    print(f"Standard Deviation of Fund: {std_daily_ret}")
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")
    print()
    print(f"Average Daily Return of Fund: {avg_daily_ret}")
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")
    print()
    print(f"Final Portfolio Value: {portvals[-1]}")


if __name__ == "__main__":
    # test_code()
    compute_portvals()