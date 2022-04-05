from datetime import datetime, timedelta
from random import random
from time import sleep

import pandas as pd
from tinkoff.invest import Client, CandleInterval

from utils.db_api import Database
from utils.db_models import OHLCV


def get_all_instruments(TOKEN):
    with Client(TOKEN) as client:
        currencies = pd.DataFrame(currency for currency in client.instruments.currencies().instruments)
        currencies["instrument_type"] = "currency"
        shares = pd.DataFrame(share for share in client.instruments.shares().instruments)
        shares["instrument_type"] = "share"
        futures = pd.DataFrame(future for future in client.instruments.futures().instruments)
        futures["instrument_type"] = "future"
        etfs = pd.DataFrame(etf for etf in client.instruments.etfs().instruments)
        etfs["instrument_type"] = "etf"
        bonds = pd.DataFrame(bond for bond in client.instruments.bonds().instruments)
        bonds["instrument_type"] = "bond"

    return pd.concat([currencies, shares, futures, etfs, bonds], axis=0)


def get_ohlcv(TOKEN, figi, i, interval=CandleInterval.CANDLE_INTERVAL_DAY):
    with Client(TOKEN) as client:
        for _ in range(10):
            try:
                all_candles = client.get_all_candles(from_=datetime.now() - timedelta(days=364 * (i + 1)),
                                                     to=datetime.now() - timedelta(days=364  * i),
                                                     interval=interval, figi=figi)
                candle_list = [candle for candle in all_candles if candle.is_complete]
                if len(candle_list) > 0:
                    r = 2 if candle_list[-1].close.units > 0 else 6
                    prices_close = [round(candle.close.units + candle.close.nano / 1_000_000_000, r) for candle in
                                    candle_list]
                    print(f"Последняя цена: {prices_close[-1]}")
                    prices_high = [round(candle.high.units + candle.close.nano / 1_000_000_000, r) for candle in
                                   candle_list]
                    prices_low = [round(candle.low.units + candle.close.nano / 1_000_000_000, r) for candle in
                                  candle_list]
                    prices_open = [round(candle.open.units + candle.close.nano / 1_000_000_000, r) for candle in
                                   candle_list]
                    volume = [candle.volume for candle in candle_list]
                    time_index = [candle.time for candle in candle_list]
                else:
                    print("Нет данных!")
                    return pd.DataFrame()

                return pd.DataFrame(
                    {"open": prices_open, "high": prices_high, "low": prices_low, "close": prices_close,
                     "volume": volume},
                    index=time_index)

            except Exception as err:
                print(err)
                sleep(random()*10)
                continue
