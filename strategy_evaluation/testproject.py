import datetime as dt
import pandas as pd
from marketsimcode import compute_portvals
import matplotlib.pyplot as plt
import ManualStrategy as mst
import StrategyLearner as strl
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import experiment1 as exp1
import experiment2 as exp2


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "cqu41"  # replace tb34 with your Georgia Tech username.


if __name__=="__main__":

    sd_in_sample = dt.datetime(2008, 1, 1)
    ed_in_sample = dt.datetime(2009, 12, 31)
    sd_out_sample = dt.datetime(2010, 1, 1)
    ed_out_sample = dt.datetime(2011, 12, 31)
    sv = 100000

    symbol_to_test="JPM"

    # In sample plot
    # generate benchmark orders
    mst = mst.ManualStrategy()
    benchmark_order = mst.benchmark_orders(symbol_to_test, sd_in_sample, ed_in_sample, sv)

    bench_vals = compute_portvals(benchmark_order, sv, 0.0, 0.0)
    normed_benchmark_vals = bench_vals / bench_vals[0]

    #calculate statistics
    port_stat=sv*normed_benchmark_vals
    cum_ret = port_stat.iloc[-1]/port_stat[0]-1
    daily_ret=port_stat/port_stat.shift(1)-1
    std_daily_ret=daily_ret.std()
    avg_daily_ret=daily_ret.mean()
    print("In sample benchmark stats: ", cum_ret, std_daily_ret,avg_daily_ret)


    #generage manual strategy
    manual_orders = mst.testPolicy(symbol_to_test, sd_in_sample, ed_in_sample, sv)

    manual_vals=compute_portvals(manual_orders, sv, 0.0, 0.0)
    normed_manual_vals = manual_vals/manual_vals[0]

    # calculate statistics
    port_stat = sv * normed_manual_vals
    cum_ret = port_stat.iloc[-1] / port_stat[0] - 1
    daily_ret = port_stat / port_stat.shift(1) - 1
    std_daily_ret = daily_ret.std()
    avg_daily_ret = daily_ret.mean()
    print("In sample Manual stats: ", cum_ret, std_daily_ret, avg_daily_ret)

    #mark entry positions
    long_entry_list = []
    short_entry_list = []

    for index, row in manual_orders.iterrows():
        if row['Long entry'] == 1:
            long_entry_list.append({'Value': 1, 'Date': index})
        if row['Short entry'] == 1:
            short_entry_list.append({'Value': 1, 'Date': index})

    long_entry = pd.DataFrame(long_entry_list).set_index('Date')
    short_entry = pd.DataFrame(short_entry_list).set_index('Date')

    plt.figure(figsize=(16, 8))
    plt.title("Manual Strategy vs Benchmark for JPM (In sample)")
    plt.xlabel("Date")
    plt.ylabel("Normalized return")

    plt.plot(normed_benchmark_vals.index, normed_benchmark_vals, label="Benchmark", color='Purple')
    plt.plot(normed_manual_vals.index, normed_manual_vals, label="Manual Strategy", color='Red')
    for index, row in long_entry.iterrows():
        plt.axvline(x=index, color='blue', linestyle='dashed')
    for index, row in short_entry.iterrows():
        plt.axvline(x=index, color='black', linestyle='dashed')
    plt.legend()
    plt.savefig("figure1.png")
    plt.clf()



    #out of sample plot
    benchmark_order = mst.benchmark_orders(symbol_to_test, sd_out_sample, ed_out_sample, sv)

    bench_vals = compute_portvals(benchmark_order, sv, 0.0, 0.0)
    normed_benchmark_vals = bench_vals / bench_vals[0]

    # calculate statistics
    port_stat = sv * normed_benchmark_vals
    cum_ret = port_stat.iloc[-1] / port_stat[0] - 1
    daily_ret = port_stat / port_stat.shift(1) - 1
    std_daily_ret = daily_ret.std()
    avg_daily_ret = daily_ret.mean()
    print("Out sample benchmark stats: ", cum_ret, std_daily_ret, avg_daily_ret)

    manual_orders = mst.testPolicy(symbol_to_test, sd_out_sample, ed_out_sample, sv)

    manual_vals = compute_portvals(manual_orders, sv, 0.0, 0.0)
    normed_manual_vals = manual_vals / manual_vals[0]

    # calculate statistics
    port_stat = sv * normed_manual_vals
    cum_ret = port_stat.iloc[-1] / port_stat[0] - 1
    daily_ret = port_stat / port_stat.shift(1) - 1
    std_daily_ret = daily_ret.std()
    avg_daily_ret = daily_ret.mean()
    print("Out sample Manual stats: ", cum_ret, std_daily_ret, avg_daily_ret)

    # mark entry positions
    long_entry_list = []
    short_entry_list = []

    for index, row in manual_orders.iterrows():
        if row['Long entry'] == 1:
            long_entry_list.append({'Value': 1, 'Date': index})
        if row['Short entry'] == 1:
            short_entry_list.append({'Value': 1, 'Date': index})

    long_entry = pd.DataFrame(long_entry_list).set_index('Date')
    short_entry = pd.DataFrame(short_entry_list).set_index('Date')

    plt.figure(figsize=(16, 8))
    plt.title("Manual Strategy vs Benchmark for JPM (Out sample)")
    plt.xlabel("Date")
    plt.ylabel("Normalized return")

    plt.plot(normed_benchmark_vals.index, normed_benchmark_vals, label="Benchmark", color='Purple')
    plt.plot(normed_manual_vals.index, normed_manual_vals, label="Manual Strategy", color='Red')
    for index, row in long_entry.iterrows():
        plt.axvline(x=index, color='blue', linestyle='dashed')
    for index, row in short_entry.iterrows():
        plt.axvline(x=index, color='black', linestyle='dashed')
    plt.legend()
    plt.savefig("figure2.png")
    plt.clf()

    exp1.plot_experiment1()
    exp2.plot_experiment2()

