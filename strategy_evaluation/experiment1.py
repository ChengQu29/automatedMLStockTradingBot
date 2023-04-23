import datetime as dt
import pandas as pd
from util import get_data
from marketsimcode import compute_portvals
import matplotlib.pyplot as plt
import ManualStrategy as mst
import StrategyLearner as strl
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def plot_experiment1():

    in_sample_sd = dt.datetime(2008, 1, 1)
    in_sample_ed = dt.datetime(2009, 12, 31)
    out_sample_sd = dt.datetime(2010, 1, 1)
    out_sample_ed = dt.datetime(2011, 12, 31)
    sv = 100000

    symbol_to_test="JPM"
    # generate benchmark orders
    man_strategy = mst.ManualStrategy()
    benchmark_order = man_strategy.benchmark_orders(symbol_to_test, in_sample_sd, in_sample_ed, sv)

    bench_vals = compute_portvals(benchmark_order, sv, 0.0, 0.0)
    normed_benchmark_vals = bench_vals / bench_vals[0]

    #generage manual strategy
    manual_orders = man_strategy.testPolicy(symbol_to_test, in_sample_sd, in_sample_ed, sv)

    manual_vals=compute_portvals(manual_orders, sv, 0.0, 0.0)
    normed_manual_vals = manual_vals/manual_vals[0]


    #generate strategy trades
    str_learner = strl.StrategyLearner()
    str_learner.add_evidence(symbol="JPM", sd=in_sample_sd, ed=in_sample_ed, sv=10000)
    trades= str_learner.testPolicy(symbol="JPM", sd=in_sample_sd, ed=in_sample_ed, sv=10000)
    strategy_orders=str_learner.generate_marketsim_orders(symbol_to_test, trades)
    strategy_vals=compute_portvals(strategy_orders, sv, 0.0, 0.0)
    normed_strategy_vals = strategy_vals/strategy_vals[0]


    plt.figure(figsize=(16, 8))
    plt.title("Manual Strategy vs Strategy Learner vs Benchmark for JPM (In sample)")
    plt.xlabel("Date")
    plt.ylabel("Normalized return")

    plt.plot(normed_benchmark_vals.index, normed_benchmark_vals, label="Benchmark", color='Purple')
    plt.plot(normed_manual_vals.index, normed_manual_vals, label="Manual Strategy", color='Green')
    plt.plot(normed_strategy_vals.index, normed_strategy_vals, label="Strategy Learner", color='Blue')
    plt.legend()
    plt.savefig("figure3.png")
    plt.clf()



    # Out of Sample
    benchmark_order = man_strategy.benchmark_orders(symbol_to_test, out_sample_sd, out_sample_ed, sv)

    bench_vals = compute_portvals(benchmark_order, sv, 0.0, 0.0)
    normed_benchmark_vals = bench_vals / bench_vals[0]

    # generage manual strategy
    manual_orders = man_strategy.testPolicy(symbol_to_test, out_sample_sd, out_sample_ed, sv)

    manual_vals = compute_portvals(manual_orders, sv, 0.0, 0.0)
    normed_manual_vals = manual_vals / manual_vals[0]

    #generate strategy trades
    str_learner.add_evidence(symbol="JPM", sd=out_sample_sd, ed=out_sample_ed, sv=10000)
    trades = str_learner.testPolicy(symbol="JPM", sd=out_sample_sd, ed=out_sample_ed, sv=10000)
    strategy_orders = str_learner.generate_marketsim_orders(symbol_to_test, trades)
    strategy_vals = compute_portvals(strategy_orders, sv, 0.0, 0.0)
    normed_strategy_vals = strategy_vals / strategy_vals[0]

    plt.figure(figsize=(16, 8))
    plt.title("Manual Strategy vs Strategy Learner vs Benchmark for JPM (Out of sample)")
    plt.xlabel("Date")
    plt.ylabel("Normalized return")

    plt.plot(normed_benchmark_vals.index, normed_benchmark_vals, label="Benchmark", color='Purple')
    plt.plot(normed_manual_vals.index, normed_manual_vals, label="Manual Strategy", color='Green')
    plt.plot(normed_strategy_vals.index, normed_strategy_vals, label="Strategy Learner", color='Blue')
    plt.legend()
    plt.savefig("figure4.png")
    plt.clf()

