from random import random
from time import sleep

import numpy as np

from utils.db_api import Database
from utils.db_models import OHLCV
from utils.tinkoff_data_downloaders import get_ohlcv
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpl_dates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplfinance as mpf


def update_daily_data(TOKEN, instruments, years):
    for figi, ticker in instruments[["figi", "ticker"]].sample(frac=1).values:
        print(f"Получаем OHLCV данные для {ticker}...")
        for i in range(years):
            data = get_ohlcv(TOKEN, figi, i)
            if not data.empty:
                data["figi"] = figi
                data["ticker"] = ticker
                data.index.name = "datetime"
                db = Database()
                db.add_records(data, OHLCV)


def support_resistance(df):
    df = df.copy()
    df["min_low"] = np.where((df["low"].rolling(5).min() == df["low"].shift(-4).rolling(5).min()) & (
            df["low"] == df["low"].rolling(5).min()), df["low"], np.nan)
    df["max_high"] = np.where((df["high"].rolling(5).max() == df["high"].shift(-4).rolling(5).max()) & (
            df["high"] == df["high"].rolling(5).max()), df["high"], np.nan)

    df["low_fractal"] = np.where((df["low"] < df["low"].shift(1)) & (df["low"].shift(1) < df["low"].shift(2))
                                 & (df["low"] < df["low"].shift(-1))
                                 & (df["low"].shift(-1) < df["low"].shift(-2)),
                                 df["low"], np.nan)
    df["high_fractal"] = np.where((df["high"] > df["high"].shift(1)) & (df["high"].shift(1) > df["high"].shift(2))
                                  & (df["high"] > df["high"].shift(-1))
                                  & (df["high"].shift(-1) > df["high"].shift(-2)),
                                  df["low"], np.nan)
    df['coalesce'] = df[["min_low", "max_high", "low_fractal", "high_fractal"]].bfill(axis=1).iloc[:, 0]
    levels = df["coalesce"]
    levels = levels.drop_duplicates().dropna().to_dict()
    result = dict()
    std = df["close"].std()
    for date, value in levels.items():
        if result.items():
            for known_date, known_value in result.items():
                if abs(value - known_value) > std:
                    result[date] = value
                else:
                    result[known_date] = (known_value + value) / 2
        else:
            result[date] = value
    return result


def plot_financial(df):
    ohlcv = df[["open", "high", "low", "close", "volume"]]

    s = mpf.make_mpf_style(base_mpl_style='seaborn', rc={'axes.grid': False})
    fig = mpf.figure(style=s, figsize=(12, 9))
    ax1 = fig.subplot()
    ax2 = ax1.twinx()
    mpf.plot(ohlcv, ax=ax1, type='ohlc', style='default', hlines=df["levels"].values)

    # mpf.plot(df, ax=ax2, type='candle', style='yahoo')
    mpf.show()

    # fig = plt.figure()
    # ax1 = plt.subplot2grid((1, 1), (0, 0))
