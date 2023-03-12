import datetime as dt

import numpy as np

import pandas as pd
from util import get_data, plot_data
from marketismcode import compute_portvals
import matplotlib.pyplot as plt

def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "cqu41"  # replace tb34 with your Georgia Tech username.


def testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009,12,31), sv = 100000):

    symbol=[symbol]

    # Get stock data
    df = get_data(symbol, pd.date_range(sd, ed))


    df=df[['JPM']]

    df['prev_day'] = df['JPM'].shift(-1)
    df['is_bigger'] = df['JPM'] > df['prev_day']
    print(df)

    plot_data(df)  # daily return indicator

    #build orders df
    orders_df  = pd.DataFrame(columns=['Symbol', 'Order', 'Shares'], index=df.index)

    net_holding=0
    for index, row in df.iterrows():

        if(row['is_bigger']):
            if net_holding==0:
                #sell
                orders_df.loc[index] = ['JPM', 'SELL', 1000]
                net_holding-=1000
            elif net_holding==1000:
                orders_df.loc[index] = ['JPM', 'SELL', 2000]
                net_holding-=2000

        else:
            if net_holding==0:
                #buy
                orders_df.loc[index] = ['JPM', 'BUY', 1000]
                net_holding+=1000
            elif net_holding==-1000:
                orders_df.loc[index] = ['JPM', 'BUY', 2000]
                net_holding+=2000


    print(orders_df)

    print(type(orders_df))
    orders_df['Symbol'] = 'JPM'
    orders_df['Order']=orders_df['Order'].fillna('BUY')
    orders_df['Shares']=orders_df['Shares'].fillna(0)
    print(orders_df)

    tos_vals = compute_portvals(orders_df, sv, 0.0, 0.0)
    normed_tos_vals = tos_vals/tos_vals[0]


    #generate benchmark orders
    benchmark_order = benchmark_orders()
    bench_vals = compute_portvals(benchmark_order, sv, 0.0 ,0.0)
    normed_benchmark_vals=bench_vals/bench_vals[0]

    plt.figure(figsize=(16,8))
    plt.title("Figure 1")
    plt.xlabel("Date")
    plt.ylabel("Profit")

    plt.plot(normed_tos_vals.index, normed_tos_vals, label="TOS")
    plt.plot(normed_benchmark_vals.index, normed_benchmark_vals, label="Benchmark")
    plt.legend()
    plt.savefig("figure1.png")
    plt.clf()


def benchmark_orders(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009,12,31), sv = 100000):

    symbols=['SPY']
    dates=pd.date_range(sd, ed)
    orders=get_data(symbols, dates=dates)

    orders['SPY']=0
    orders.rename(columns={'SPY': 'Symbol'}, inplace=True)
    orders['Order']=0
    orders['Shares']=0

    orders['Symbol']='JPM'
    orders['Order'] = 'BUY'
    orders.iloc[0,2] = 1000

    #return orders dataframe
    return orders


def compute_daily_returns(df):
    """Compute and return the daily return values."""
    # Quiz: Your code here
    # Note: Returned DataFrame must have the same number of rows
    daily_returns = df.copy()
    daily_returns[1:] = (df[1:] / df[:-1].values) - 1 # compute daily returns for row 1 onwards
    #daily_returns = (df / df.shift(1)) - 1  # much easier with Pandas!
    daily_returns.iloc[0, :] = 0  # Pandas leaves the 0th row full of Nans
    return daily_returns

if __name__ == "__main__":
    # base_policy(symbol='JPM')
    testPolicy()