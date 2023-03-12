import datetime as dt

import numpy as np

import pandas as pd
from util import get_data, plot_data
import TheoreticallyOptimalStrategy as tos

df_trades = tos.testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009,12,31), sv=100000)
