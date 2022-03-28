from random import random
from time import sleep

from utils.db_api import Database
from utils.db_models import OHLCV
from utils.tinkoff_data_downloaders import get_ohlcv


def update_daily_data(TOKEN, instruments):
    for figi, ticker in instruments[["figi", "ticker"]].sample(frac=1).values:
        print(f"Получаем OHLCV данные для {ticker}...")
        for i in range(20):
            data = get_ohlcv(TOKEN, figi, i)
            if not data.empty:
                data["figi"] = figi
                data["ticker"] = ticker
                data.index.name = "datetime"
                db = Database()
                db.add_records(data, OHLCV)
        sleep(random())
