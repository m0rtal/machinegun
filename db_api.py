import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import exc
from sqlalchemy.orm import sessionmaker

from utils import db_models
from utils.db_models import OHLCV


class Database:
    def __init__(self, db_url="sqlite:///utils/database.sqlite"):
        self.engine = create_engine(db_url, pool_pre_ping=True, echo=False)
        db_models.Base.metadata.create_all(bind=self.engine)
        self.maker = sessionmaker(bind=self.engine)
        self.connection = self.engine.connect()

    def add_records(self, df, model):
        session = self.maker()
        payload = df.reset_index().to_dict(orient="records")
        for load in payload:
            data = model(**load)
            session.add(data)
            try:
                session.commit()
                session.close()
            except exc.IntegrityError as err:
                # print(err)
                session.rollback()
                session.close()

    def get_ohlcv_data_by_figi(self, figi):
        session = self.maker()
        query = session.query(OHLCV).filter(OHLCV.figi == figi).order_by(OHLCV.figi, OHLCV.datetime)
        result = pd.read_sql(query.statement, session.bind)
        return result

