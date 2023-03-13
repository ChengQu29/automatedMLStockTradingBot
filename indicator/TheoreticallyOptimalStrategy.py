import datetime as dt
import pandas as pd
from util import get_data
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

    #make a column for the previous price
    df['next_day'] = df['JPM'].shift(-1)
    df['is_bigger'] = df['JPM'] > df['next_day']
    # print(df)

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


    # print(orders_df)
    #
    # print(type(orders_df))
    orders_df['Symbol'] = 'JPM'
    orders_df['Order']=orders_df['Order'].fillna('BUY')
    orders_df['Shares']=orders_df['Shares'].fillna(0)
    # print(orders_df)

    tos_vals = compute_portvals(orders_df, sv, 0.0, 0.0)
    normed_tos_vals = tos_vals/tos_vals[0]


    #generate benchmark orders
    benchmark_order = benchmark_orders()
    bench_vals = compute_portvals(benchmark_order, sv, 0.0 ,0.0)
    normed_benchmark_vals=bench_vals/bench_vals[0]

    plt.figure(figsize=(16,8))
    plt.title("Theoretically Optimal Strategy(TOS) vs Benchmark for JPM")
    plt.xlabel("Date")
    plt.ylabel("Normalized return")

    plt.plot(normed_tos_vals.index, normed_tos_vals, label="TOS", color='red')
    plt.plot(normed_benchmark_vals.index, normed_benchmark_vals, label="Benchmark", color='Purple')
    plt.legend()
    plt.savefig("figure1.png")
    plt.clf()

    '''
    print statistics
    '''
    daily_ret_tos= tos_vals/tos_vals.shift(1)-1
    daily_ret_tos=daily_ret_tos[1:]
    cum_ret_tos = (tos_vals[-1] / tos_vals[0] - 1)
    mean_daily_ret_tos = daily_ret_tos.mean().round(6)
    std_daily_ret_tos = daily_ret_tos.std().round(6)
    print("TOS cumulative return: ", cum_ret_tos.round(6))
    print("TOS mean of daily return: ", mean_daily_ret_tos)
    print('TOS standard deviation of daily returns: ', std_daily_ret_tos)

    daily_ret_bench = bench_vals / bench_vals.shift(1) - 1
    daily_ret_bench = daily_ret_bench[1:]
    cum_ret_bench = (bench_vals[-1] / bench_vals[0] - 1)
    mean_daily_ret_bench = daily_ret_bench.mean().round(6)
    std_daily_ret_bench = daily_ret_bench.std().round(6)
    print("Bench cumulative return: ", cum_ret_bench.round(6))
    print("Bench mean of daily return: ", mean_daily_ret_bench)
    print('Bench standard deviation of daily returns: ', std_daily_ret_bench)


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


if __name__ == "__main__":
    # base_policy(symbol='JPM')
    testPolicy()