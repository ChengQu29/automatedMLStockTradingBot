import datetime as dt
import pandas as pd
from util import get_data
import matplotlib.pyplot as plt

def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "cqu41"  # replace tb34 with your Georgia Tech username.



def MACD(normed_price):
    ema_12 = normed_price.ewm(span=12, min_periods=12, adjust=False).mean()
    ema_26 = normed_price.ewm(span=26, min_periods=26, adjust=False).mean()

    MACD_line=ema_12-ema_26
    signal_line=MACD_line.ewm(span=9, min_periods=9, adjust=False).mean()

    return MACD_line, signal_line



# Average Directional Index
def BB(lookback, normed_price):

    rolling_mean=normed_price.rolling(window=lookback).mean()

    rolling_std=normed_price.rolling(window=lookback).std()

    upper_band=rolling_mean+rolling_std*2
    lower_band=rolling_mean-rolling_std*2

    bb=(normed_price-rolling_mean)/(rolling_std*2)
    return upper_band, lower_band, bb

def momentum(lookback, normed_price):

    return normed_price / normed_price.shift(lookback) - 1

def print_plot(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009,12,31), lookback=20):

    symbol=[symbol]

    # Get stock data
    price_df = get_data(symbol, pd.date_range(sd, ed))
    price_df=price_df[["JPM"]]

    normed_price_df=price_df/price_df.iloc[0]

    # print(price_df)

    sma=normed_price_df.rolling(window=lookback).mean()

    #sma ratio from lecture
    sma_ratio=normed_price_df/sma

    plt.figure(figsize=(16, 8))
    plt.title("Price and SMA of JPM (lookback = 20)")
    plt.xlabel("Date")

    plt.plot(sma, label="SMA")
    plt.plot(normed_price_df, label="JPM")

    plt.legend()
    plt.savefig("figure3.png")
    plt.clf()

    plt.figure(figsize=(16, 8))
    plt.title("Price/SMA ratio of JPM (lookback =20)")
    plt.xlabel("Date")
    plt.ylabel("Price/SMA ratio")
    plt.plot(sma_ratio, label="Price/SMA Ratio")
    plt.axhline(y=1.1, linestyle='--')
    plt.axhline(y=0.9, linestyle='--')
    plt.legend()
    plt.savefig("figure4.png")
    plt.clf()
    momentum_df = momentum(lookback, normed_price_df)

    plt.figure(figsize=(16, 8))
    plt.title("Momentum line of JPM (20 days)")
    plt.xlabel("Date")
    plt.ylabel("Momentum")
    plt.plot(momentum_df, label="Momentum")

    plt.axhline(y=0.5, linestyle='--')
    plt.axhline(y=-0.5, linestyle='--')
    plt.legend()
    plt.savefig("figure2.png")
    plt.clf()

    #BB
    upper_band, lower_band, bb= BB(lookback, normed_price_df)
    plt.figure(figsize=(16, 8))
    plt.title("Bollinger Band")
    plt.xlabel("Date")
    plt.ylabel("BB")
    plt.plot(normed_price_df, label="Price of JPM (normalized)")
    plt.plot(upper_band, label="Upper band")
    plt.plot(lower_band, label="lower band")
    plt.legend()
    plt.savefig("figure5.png")
    plt.clf()

    #BB quantative
    plt.figure(figsize=(16, 8))
    plt.title("Bollinger Band Percentage")
    plt.xlabel("Date")
    plt.ylabel("BB%")
    plt.plot(bb, label="BB percentage")
    plt.axhline(y=1.0, linestyle='--')
    plt.axhline(y=-1.0, linestyle='--')
    plt.legend()
    plt.savefig("figure6.png")
    plt.clf()

    # MACD
    plt.figure(figsize=(16, 8))
    plt.title("MACD")
    plt.xlabel("Date")
    macd_line, signal_line=MACD(normed_price_df)
    plt.plot(macd_line, label="MACD")
    plt.plot(signal_line, label="Signal line")
    plt.axhline(y=0.0, label="Zero line")
    plt.legend()
    plt.savefig("figure7.png")
    plt.clf()

    #EMA
    ema_50=normed_price_df.ewm(span=50, min_periods=0, adjust=False).mean()
    ema_200 = normed_price_df.ewm(span=200, min_periods=0, adjust=False).mean()
    plt.figure(figsize=(16, 8))
    plt.title("EMA")
    plt.xlabel("Date")
    plt.plot(ema_50, label="EMA 50")
    plt.plot(ema_200, label="EMA 200")
    plt.plot(normed_price_df, label="Price (normalized)")
    plt.legend()
    plt.savefig("figure8.png")
    plt.clf()
