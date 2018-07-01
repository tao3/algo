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
    # define the windows for the moving averages
    short_window = 22
    long_window = 42

    # Skip as many bars as long_window to properly compute the average
    context.i += 1
    if context.i < long_window:
        return


    short_data = context.alldata.loc[data.current_dt - pd.to_timedelta(np.arange(short_window),'T')].sort_index()[:-1]
    sma20 = short_data.tail(20).price.mean()
    prev_sma20 = short_data.head(20).price.mean()

    long_data = context.alldata.loc[data.current_dt - pd.to_timedelta(np.arange(long_window),'T')].sort_index()[:-1]
    sma40 = long_data.tail(40).price.mean()
    prev_sma40 = long_data.head(40).price.mean()

    rsi = ta.RSI(short_data.price.values, timeperiod=16)[-1]
    cci = ta.CCI(short_data.high.values, short_data.low.values, short_data.close.values, timeperiod=5)[-1]
    last_tick = short_data.tail(1)

    price = context.alldata.loc[data.current_dt].price

    # If base_price is not set, we use the current value. This is the
    # price at the first bar which we reference to calculate price_change.
    if context.base_price is None:
        context.base_price = price
    price_change = (price - context.base_price) / context.base_price

    # Save values for later inspection
    record(price=price,
           cash=context.portfolio.cash,
           price_change=price_change,
           sma20=sma20,
           prev_sma20=prev_sma20,
           sma40=sma40,
           prev_sma40=prev_sma40,
           cci=cci,
           rsi=rsi,
           stop_price=context.stop_price,
           )



    # Since we are using limit orders, some orders may not execute immediately
    # we wait until all orders are executed before considering more trades.
    orders = context.blotter.open_orders
    if len(orders) > 0:
        return

    # Exit if we cannot trade
    if not data.can_trade(context.asset):
        return

    # We check what's our position on our portfolio and trade accordingly
    pos_amount = context.portfolio.positions[context.asset].amount

    # Trading logic
    #1 both 20 and 40 sma are sloping upwards
    #2 20 sma is above 40 sma
    #3 CCI < -100: oversold
    #4 low of the candle below 20sma
    #5 close of the candle above 40sma
    set_trailing_stop(context,data)
    buy_signal = (sma20>prev_sma20) and (sma40 > prev_sma40) and (sma20>sma40) and \
                 (cci < -100) and \
                 (last_tick.low.values[0] < sma20) and (last_tick.close.values[0] > sma40) and ((price - last_tick.high.values[0])/last_tick.high.values[0]>0.0010)

    sell_signal = (price < context.stop_price) or (price < context.trailing_stop_price)

    if buy_signal and pos_amount == 0:
        # we buy 100% of our portfolio for this asset
        order_target_percent(context.asset, 1)

    elif sell_signal and pos_amount > 0:
        # we sell all our positions for this asset
        order_target_percent(context.asset, 0)
        context.stop_price = 0
        context.trailing_stop_price = 0


def set_trailing_stop(context, data):
    if context.portfolio.positions[context.asset].amount > 0:
        price = data.current(context.asset, 'price')

        context.stop_price = context.portfolio.positions[context.asset].cost_basis*context.stop_pct
        context.trailing_stop_price = max(context.trailing_stop_price, price*context.trailing_stop_pct)




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
    perf.loc[:, ['price', 'sma20', 'sma40']].plot(
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

    # Fourth chart: Plot our cash
    ax4 = plt.subplot(414, sharex=ax1)
    perf.loc[:,['cci', 'rsi']].plot(ax=ax4)
    ax4.set_ylabel('cci\n({})'.format(base_currency))
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
        start=pd.to_datetime('2017-01-03', utc=True),
        end=pd.to_datetime('2017-05-03', utc=True),
    )