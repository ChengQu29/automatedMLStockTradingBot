import datetime as dt
import pandas as pd
from util import get_data
from marketsim import compute_portvals
import matplotlib.pyplot as plt
import ManualStrategy as mst
import StrategyLearner as strl
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

if __name__=="__main__":

    sd_in_sample = dt.datetime(2008, 1, 1)
    ed_in_sample = dt.datetime(2009, 12, 31)
    sd_out_sample = dt.datetime(2010, 1, 1)
    ed_out_sample = dt.datetime(2011, 12, 31)
    sv = 100000

    symbol_to_test="JPM"
    # generate benchmark orders
    mst = mst.ManualStrategy()
    benchmark_order = mst.benchmark_orders(symbol_to_test, sd_in_sample, ed_in_sample, sv)

    bench_vals = compute_portvals(benchmark_order, sv, 0.0, 0.0)
    normed_benchmark_vals = bench_vals / bench_vals[0]

    #generage manual strategy
    manual_orders = mst.testPolicy(symbol_to_test, sd_in_sample, ed_in_sample, sv)

    manual_vals=compute_portvals(manual_orders, sv, 0.0, 0.0)
    normed_manual_vals = manual_vals/manual_vals[0]


    plt.figure(figsize=(16, 8))
    plt.title("Manual Strategy vs Benchmark for JPM (In sample)")
    plt.xlabel("Date")
    plt.ylabel("Normalized return")

    plt.plot(normed_benchmark_vals.index, normed_benchmark_vals, label="Benchmark", color='Purple')
    plt.plot(normed_manual_vals.index, normed_manual_vals, label="Manual Strategy", color='Green')
    plt.legend()
    plt.savefig("figure1.png")
    plt.clf()

    benchmark_order = mst.benchmark_orders(symbol_to_test, sd_out_sample, ed_out_sample, sv)

    bench_vals = compute_portvals(benchmark_order, sv, 0.0, 0.0)
    normed_benchmark_vals = bench_vals / bench_vals[0]

    # generage manual strategy
    manual_orders = mst.testPolicy(symbol_to_test, sd_out_sample, ed_out_sample, sv)

    manual_vals = compute_portvals(manual_orders, sv, 0.0, 0.0)
    normed_manual_vals = manual_vals / manual_vals[0]

    plt.figure(figsize=(16, 8))
    plt.title("Manual Strategy vs Benchmark for JPM (Out sample)")
    plt.xlabel("Date")
    plt.ylabel("Normalized return")

    plt.plot(normed_benchmark_vals.index, normed_benchmark_vals, label="Benchmark", color='Purple')
    plt.plot(normed_manual_vals.index, normed_manual_vals, label="Manual Strategy", color='Green')
    plt.legend()
    plt.savefig("figure2.png")
    plt.clf()

