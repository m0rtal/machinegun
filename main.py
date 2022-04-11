# https://github.com/Tinkoff/invest-python/tree/main/examples
# https://charlieoneill.medium.com/predicting-the-price-of-bitcoin-with-multivariate-pytorch-lstms-695bc294130
# https://medium.com/geekculture/implementing-the-most-popular-indicator-on-tradingview-using-python-239d579412ab

import os
import seaborn as sns

import dotenv
import numpy as np
import pandas as pd
import plotly
import torch
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression
from tinkoff.invest import Client
from torch import nn
from torch.autograd import Variable

from utils.misc import update_daily_data, support_resistance, plot_financial
from utils.tinkoff_data_downloaders import get_all_instruments
from utils.db_api import Database
import matplotlib.dates as mpl_dates


import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

dotenv.load_dotenv(".env")
TOKEN = os.getenv("INVEST_TOKEN")


def convert_units(x):
    return round(x.units + x.nano / 1_000_000_000, 2)


def main(TOKEN) -> None:
    CONTRACT_PREFIX = "tinkoff.public.invest.api.contract.v1."
    with Client(TOKEN) as client:
        accounts = client.users.get_accounts()
        print("\nСписок текущих аккаунтов\n")
        for account in accounts.accounts:
            print("\t", account.id, account.name, account.access_level.name)

        print("\nИнформация\n")
        info = client.users.get_info()
        print(info)


if __name__ == '__main__':
    dotenv.load_dotenv(".env")
    TOKEN = os.getenv("INVEST_TOKEN")
    # PROXY = os.getenv("PROXY")

    # os.environ['http_proxy'] = PROXY
    # os.environ['HTTP_PROXY'] = PROXY

    # таблица всех инструментов
    all_instruments = get_all_instruments(TOKEN)
    # обновим дневные данные
    # update_daily_data(TOKEN, all_instruments, years=1)

    shares = all_instruments[all_instruments["instrument_type"] == "share"]

    db = Database()

    for figi in shares["figi"].to_list():
        df = db.get_ohlcv_data_by_figi(figi)
        channel_length = max(-200, -len(df))
        crop = max(-600, -len(df))
        model = LinearRegression()
        model.fit(df.index.values[channel_length:, np.newaxis], df['close'][channel_length:])
        df['channel_mid'] = np.nan
        df['channel_mid'].iloc[channel_length:] = model.predict(df.index.values[channel_length:, np.newaxis])
        # df['channel_high'] = df['channel_mid'] + df['close'].iloc[channel_length:].std()
        # df['channel_low'] = df['channel_mid'] - df['close'].iloc[channel_length:].std()

        n = 10  # number of points to be checked before and after

        # df.dropna(axis=0, inplace=True)

        # Find local peaks
        df['min'] = df.iloc[argrelextrema(df['low'].values, np.less_equal, order=n)[0]]['low']
        df['max'] = df.iloc[argrelextrema(df['high'].values, np.greater_equal, order=n)[0]]['high']

        df_min = df['min'][channel_length:].dropna()
        model = LinearRegression()
        model.fit(df_min.index.values[:, np.newaxis], df_min)
        df['mins_line'] = np.nan
        df['mins_line'].iloc[channel_length:] = model.predict(df.index.values[channel_length:, np.newaxis])

        df_max = df['max'][channel_length:].dropna()
        model = LinearRegression()
        model.fit(df_max.index.values[:, np.newaxis], df_max)
        df['maxs_line'] = np.nan
        df['maxs_line'].iloc[channel_length:] = model.predict(df.index.values[channel_length:, np.newaxis])

        # df = add_all_ta_features(prices_df, open="open", high="high", low="low", close="close", volume="volume",
        #                          fillna=False)

        # df["date"] = df["datetime"].dt.date
        df.set_index('datetime', inplace=True)
        df["YM"] = df.index.strftime("%Y-%m")
        df["month_open"] = df.groupby("YM")["open"].head(1)
        df["month_open"] = df["month_open"].fillna(method="ffill")
        # prices_monthly = df["close"].resample("M").ohlc()
        df["monthly_std"] = df["close"].resample("M").mean().rolling(14).std().resample("d").mean()
        df = df[crop:]

        # https://medium.com/swlh/how-to-analyze-volume-profiles-with-python-3166bb10ff24

        fig, ax1 = plt.subplots(figsize=(20, 10))
        ax1.plot(df["close"], color="green")
        ax1.plot(df["channel_mid"])

        levels = support_resistance(df)

        for date, level in levels.items():
            plt.hlines(level, xmin=date, xmax=max(df.index), color="gray")

        date_format = mpl_dates.DateFormatter('%d %b %Y')
        ax1.xaxis.set_major_formatter(date_format)
        fig.show()
        fig.savefig("test.png")

        break

        # df.drop(['volume', 'open', 'high', 'low', 'min', 'max'], axis=1).plot()
        # # plt.scatter(df.index, df['min'], c='g')
        # # plt.scatter(df.index, df['max'], c='b')
        # plt.show()
        1

    # # main(TOKEN)
    #
    # with Client(TOKEN) as client:
    #     accounts = client.users.get_accounts()
    #     iis_id = ''.join(account.id for account in accounts.accounts if account.name == "ИИС")
    #     margins = client.users.get_margin_attributes(account_id=iis_id)
    #     schedule = (client.instruments.trading_schedules(exchange='MOEX_PLUS', from_=datetime.today(),
    #                                                      to=datetime.today())).exchanges[0].days[0]
    #     print(
    #         f"Начало торгов: {schedule.start_time}\n"
    #         f"Текущее время: {datetime.now(pytz.utc)}\n"
    #         f"Конец торгов: {schedule.end_time}")
    #     trade_allowed = margins.liquid_portfolio.units >= margins.starting_margin.units \
    #                     and schedule.is_trading_day \
    #                     and schedule.start_time <= datetime.now(pytz.utc) \
    #                     and schedule.end_time >= datetime.now(pytz.utc)
    #
    #     print(f'Торговля разрешена: {trade_allowed}')
    #     print(f"Фин.средств доступно: {margins.liquid_portfolio.units} {margins.liquid_portfolio.currency}")
    #
    #     # portfolio = client.operations.get_portfolio(account_id=iis_id)
    #     # print(*portfolio.positions, sep='\n')
    #
    #     # positions = client.operations.get_positions(account_id=iis_id)
    #     # print(positions)
    #
    #
    #     # instruments = shares + futures + etfs
    #     # instruments = [i for i in instruments if i.api_trade_available_flag]
    #
    #     # delta zero strategy
    #     zero_df = pd.merge(futures_df, shares_df, left_on="basic_asset", right_on="ticker", how="inner",
    #                        suffixes=('_future', '_share'))
    #     svod = pd.DataFrame(
    #         columns=["share_price", "share_name", "share_figi", "future_price", "future_name", "future_figi",
    #                  "price_diff", "money_needed", "days_till", "monthly_income_money", "monthly_p"])
    #     for f_f, sh_f in zip(zero_df["figi_future"].to_list(), zero_df["figi_share"].to_list()):
    #         future = client.instruments.future_by(id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_FIGI, id=f_f)
    #         share = client.instruments.share_by(id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_FIGI, id=sh_f)
    #         future_price = convert_units(
    #             client.market_data.get_last_prices(figi=[f_f]).last_prices[0].price) * future.instrument.lot
    #         share_price = convert_units(
    #             client.market_data.get_last_prices(figi=[sh_f]).last_prices[0].price) * share.instrument.lot
    #         if future_price / share_price > 2:
    #             share_price *= 10
    #         if future_price / share_price > 2:
    #             share_price *= 10
    #
    #         print("=" * 200)
    #         # print(f"Фьючерс {future.instrument.name}, цена {future_price}")
    #         # print(f"Акция {share.instrument.name}, цена {share_price}")
    #
    #         days_till_expiration = (future.instrument.last_trade_date.date() - datetime.utcnow().date()).days
    #         # print(f"Дней до экспирации: {days_till_expiration}")
    #         diff = abs(future_price - share_price)
    #         # print(f"Соотношение фьючерс/акция: {diff}")
    #
    #         profitability = round(diff / (days_till_expiration / 30), 2)
    #         percent = round(profitability / (future_price + share_price) * 100, 2)
    #         # print(
    #         #     f"Вложенные средства: {future_price + share_price} и месячная доходность: "
    #         #     f"{profitability} ({percent}%)\n")
    #
    #         payload = {
    #             "share_price": share_price,
    #             "share_name": share.instrument.name,
    #             "share_figi": share.instrument.figi,
    #             "future_price": future_price,
    #             "future_name": future.instrument.name,
    #             "future_figi": future.instrument.figi,
    #             "price_diff": diff,
    #             "money_needed": future_price + share_price,
    #             "days_till": days_till_expiration,
    #             "monthly_income_money": profitability,
    #             "monthly_p": percent
    #         }
    #         payload = pd.DataFrame(payload, index=[0])
    #         svod = pd.concat([svod, payload], ignore_index=True, join="outer", axis=0)
    #         svod.sort_values(axis=0, by="monthly_p", ascending=False, inplace=True)
    #         with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width',
    #                                1000):  # more options can be specified also
    #             print(svod)
    #         sleep(2)
    #     1
    #     # for instrument in instruments:
    #     #     print(instrument)
    #     #     print(f"Работаем с {instrument.name}, тикер {instrument.ticker}, figi {instrument.figi}")
    #     #     print(f"API trade available: {instrument.api_trade_available_flag and trade_allowed}\n"
    #     #           f"Buy available: {instrument.buy_available_flag and trade_allowed}\n"
    #     #           f"Sell available: {instrument.sell_available_flag and trade_allowed}\n"
    #     #           f"Trading status: {instrument.trading_status}")
    #     #
    #     #     prices_df = get_ohlcv(instrument.figi)
    #     #
    #     #     for column in ("open", "high", "low", "close"):
    #     #         if instrument.currency == 'usd':
    #     #             prices_df[column] = prices_df[column].multiply(usd_rub[column], level=0) * instrument.lot
    #     #         elif instrument.currency == 'eur':
    #     #             prices_df[column] = prices_df[column].multiply(eur_rub[column], level=0) * instrument.lot
    #     #         else:
    #     #             prices_df[column] = prices_df[column] * instrument.lot
    #     #

    #     #     plt.plot(prices_df.index, prices_df['close'])
    #     #     # plt.plot(prices_df.index, prices_df['volume'])
    #     #     plt.legend(loc="best")
    #     #     plt.title(instrument.name)
    #     #     plt.show()
