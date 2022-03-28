from sqlalchemy import (
    Column,
    Integer,
    BigInteger,
    Float,
    String,
    ForeignKey,
    Date,
    DateTime

)
from sqlalchemy.ext.declarative import declarative_base  # класс, от которого будем наследовать все таблицы БД
from sqlalchemy.orm import relationship

Base = declarative_base()


class OHLCV(Base):
    __tablename__ = "OHLCV"
    datetime = Column(DateTime, nullable=False, unique=False, index=True, primary_key=True)
    figi = Column(String(10), nullable=False, unique=False, index=True, primary_key=True)
    ticker = Column(String(10), nullable=False, unique=False, index=True, primary_key=True)
    open = Column(Float, nullable=False, unique=False)
    high = Column(Float, nullable=False, unique=False)
    low = Column(Float, nullable=False, unique=False)
    close = Column(Float, nullable=False, unique=False)
    volume = Column(BigInteger, nullable=False, unique=False)

    # baseinfo = relationship("BaseData", backref="DailyData")
    # rates = relationship("Rates", backref="DailyData")
    # quarterly = relationship("QuarterlyData", backref="DailyData")
