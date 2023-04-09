import datetime as dt
import pandas as pd
from util import get_data
from marketsim import compute_portvals
import matplotlib.pyplot as plt
import ManualStrategy as mst
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

if __name__=="__main__":

    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    sv = 100000

    symbol_to_test="JPM"
    # generate benchmark orders
    mst = mst.ManualStrategy()
    benchmark_order = mst.benchmark_orders(symbol_to_test)

    bench_vals = compute_portvals(benchmark_order, sv, 0.0, 0.0)
    normed_benchmark_vals = bench_vals / bench_vals[0]

    #generage manual strategy
    manual_orders = mst.testPolicy(symbol_to_test, sd, ed, sv)

    manual_vals=compute_portvals(manual_orders, sv, 0.0, 0.0)
    normed_manual_vals = manual_vals/manual_vals[0]


    plt.figure(figsize=(16, 8))
    plt.title("Manual Strategy vs Benchmark for JPM")
    plt.xlabel("Date")
    plt.ylabel("Normalized return")

    plt.plot(normed_benchmark_vals.index, normed_benchmark_vals, label="Benchmark", color='Purple')
    plt.plot(normed_manual_vals.index, normed_manual_vals, label="Manual Strategy", color='Green')
    plt.legend()
    plt.savefig("figure1.png")
    plt.clf()

