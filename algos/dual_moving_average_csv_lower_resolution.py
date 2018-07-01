import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from logbook import Logger
import talib as ta

from catalyst import run_algorithm
from catalyst.api import (record, symbol, order_target_percent, )
from catalyst.exchange.utils.stats_utils import extract_transactions

NAMESPACE = 'dual_moving_average'
log = Logger(NAMESPACE)
DATA_PATH = "/Users/taoranli/Documents/python/cryptotrading/enigma-catalyst/data/"
FILE_NAME = DATA_PATH + "bitfinex_minute_2016-01-01_2018-05-01.csv"

def initialize(context):
    context.i = 0
    context.asset = symbol('btc_usd')
    context.base_price = None
    context.stop_price = 0
    context.stop_pct = 0.99
    context.trailing_stop_price = 0
    context.trailing_stop_pct = 0.98
    alldata = pd.read_csv(FILE_NAME)
    alldata = alldata.set_index(pd.DatetimeIndex(alldata['Time']))
    context.alldata = alldata



def handle_data(context, data):
    context.i += 1
    short_window = 20
    long_window = 60
    minutes = 60
    minutes_str = '60T'

    now = data.current_dt

    if context.i < long_window*minutes:
        return

    if not context.i % minutes:
        hist_data = context.alldata.loc[now - pd.to_timedelta(np.arange(long_window * minutes),'T')].sort_index()[:-1]
        resample_hist_data = hist_data.resample(minutes_str).agg({'price': 'mean', 'volume': 'sum', 'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})

        sma_short= resample_hist_data[-short_window:].price.mean()
        sma_long = resample_hist_data.price.mean()
        rsi = ta.RSI(resample_hist_data.price.values, timeperiod=14)[-1]

        price = context.alldata.loc[now].price

        if context.base_price is None:
            context.base_price = price
        price_change = (price - context.base_price) / context.base_price

        record(price=price,
               cash=context.portfolio.cash,
               price_change=price_change,
               sma_short=sma_short,
               sma_long=sma_long,
               rsi=rsi)




def analyze(context, perf):
    # Get the base_currency that was passed as a parameter to the simulation
    exchange = list(context.exchanges.values())[0]
    base_currency = exchange.base_currency.upper()

    # First chart: Plot portfolio value using base_currency
    ax1 = plt.subplot(411)
    perf.loc[:, ['portfolio_value']].plot(ax=ax1)
    ax1.legend_.remove()
    ax1.set_ylabel('Portfolio Value\n({})'.format(base_currency))
    start, end = ax1.get_ylim()
    ax1.yaxis.set_ticks(np.arange(start, end, (end - start) / 5))

    # Second chart: Plot asset price, moving averages and buys/sells
    ax2 = plt.subplot(412, sharex=ax1)
    perf.loc[:, ['price', 'sma_short', 'sma_long']].plot(
        ax=ax2,
        label='Price')

    ax2.set_ylabel('{asset}\n({base})'.format(
        asset=context.asset.symbol,
        base=base_currency
    ))
    start, end = ax2.get_ylim()
    ax2.yaxis.set_ticks(np.arange(start, end, (end - start) / 5))

    transaction_df = extract_transactions(perf)
    if not transaction_df.empty:
        buy_df = transaction_df[transaction_df['amount'] > 0]
        sell_df = transaction_df[transaction_df['amount'] < 0]
        ax2.scatter(
            buy_df.index.to_pydatetime(),
            perf.loc[buy_df.index, 'price'],
            marker='^',
            s=100,
            c='green',
            label=''
        )
        ax2.scatter(
            sell_df.index.to_pydatetime(),
            perf.loc[sell_df.index, 'price'],
            marker='v',
            s=100,
            c='red',
            label=''
        )

    # Third chart: Compare percentage change between our portfolio
    # and the price of the asset
    ax3 = plt.subplot(413, sharex=ax1)
    perf.loc[:, ['algorithm_period_return', 'price_change']].plot(ax=ax3)
    ax3.legend_.remove()
    ax3.set_ylabel('Percent Change')
    start, end = ax3.get_ylim()
    ax3.yaxis.set_ticks(np.arange(start, end, (end - start) / 5))

    # Fourth chart: Plot our indicator
    ax4 = plt.subplot(414, sharex=ax1)
    perf.loc[:, ['rsi']].plot(ax=ax4)
    ax4.set_ylabel('rsi\n({})'.format(base_currency))
    start, end = ax4.get_ylim()
    ax4.yaxis.set_ticks(np.arange(0, end, end / 5))

    plt.show()


if __name__ == '__main__':
    run_algorithm(
        capital_base=1000,
        data_frequency='minute',
        initialize=initialize,
        handle_data=handle_data,
        analyze=analyze,
        exchange_name='bitfinex',
        algo_namespace=NAMESPACE,
        base_currency='usd',
        start=pd.to_datetime('2017-01-03 00:00:00', utc=True),
        end=pd.to_datetime('2017-01-30 00:00:00', utc=True),
    )