import datetime as dt
import pandas as pd
from util import get_data
from marketsimcode import compute_portvals
import matplotlib.pyplot as plt
import ManualStrategy as mst
import StrategyLearner as strl
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import numpy as np

def plot_experiment2():

    in_sample_sd = dt.datetime(2008, 1, 1)
    in_sample_ed = dt.datetime(2009, 12, 31)
    sv = 100000

    symbol_to_test="JPM"

    cum_returns = []
    std_daily_rets = []
    avg_daily_rets = []
    num_of_orders = []

    start = 0.005  # 0.5%
    stop = 0.095  # 9.5%
    num = 10
    impacts = np.linspace(start, stop, num)

    for impact in impacts:
        str_learner = strl.StrategyLearner(impact=impact)
        str_learner.add_evidence(symbol="JPM", sd=in_sample_sd, ed=in_sample_ed, sv=10000)
        trades= str_learner.testPolicy(symbol="JPM", sd=in_sample_sd, ed=in_sample_ed, sv=10000)
        strategy_orders = str_learner.generate_marketsim_orders(symbol_to_test, trades)
        str_vals=compute_portvals(strategy_orders, sv, 0, 0)
        normed_str_vals=str_vals/str_vals[0]


        # calculate statistics
        num_of_orders.append(strategy_orders.shape[0])
        port_stat = sv * normed_str_vals
        cum_ret = port_stat.iloc[-1] / port_stat[0] - 1
        daily_ret = port_stat / port_stat.shift(1) - 1
        std_daily_ret = daily_ret.std()
        avg_daily_ret = daily_ret.mean()


        cum_returns.append(cum_ret)
        std_daily_rets.append(std_daily_ret)
        avg_daily_rets.append(avg_daily_ret)

    plt.figure(figsize=(16, 8))
    plt.title("Effect of Impact (In sample)")
    plt.xlabel("Impact")
    plt.ylabel("Normalized return")
    plt.plot(impacts, cum_returns, label="Normalized return", color='Purple')
    plt.legend()
    plt.savefig("figure5.png")
    plt.clf()

    plt.figure(figsize=(16, 8))
    plt.title("Effect of Impact (In sample)")
    plt.xlabel("Impact")
    plt.ylabel("Number of trades executed")
    plt.plot(impacts, num_of_orders, label="Number of trades executed", color='Purple')
    plt.legend()
    plt.savefig("figure6.png")
    plt.clf()
