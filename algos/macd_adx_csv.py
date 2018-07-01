import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from logbook import Logger
import talib as ta
import os

from catalyst import run_algorithm
from catalyst.api import (record, symbol, order_target_percent, )
from catalyst.exchange.utils.stats_utils import extract_transactions

NAMESPACE = 'dual_moving_average'
log = Logger(NAMESPACE)
#DATA_PATH = "/Users/taoranli/Documents/python/cryptotrading/enigma-catalyst/data/"
#FILE_NAME = DATA_PATH + "bitfinex_minute_2016-01-01_2018-05-01.csv"
FILE_NAME = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/bitfinex_minute_2016-01-01_2018-05-01.csv")

def initialize(context):
    context.i = 0
    context.asset = symbol('btc_usd')
    context.base_price = None
    context.stop_price = 0
    context.stop_pct = 0.97
    context.trailing_stop_price = 0
    context.trailing_stop_pct = 0.97
    context.profit_taking_pct = 1.12
    context.oversold_state = 0
    context.oversold_time = pd.to_datetime('2017-01-01 00:00:00',utc=True)
    context.lastcrossover_time = pd.to_datetime('2017-01-01 00:00:00', utc=True)
    alldata = pd.read_csv(FILE_NAME)
    alldata = alldata.set_index(pd.DatetimeIndex(alldata['Time']))
    context.alldata = alldata



def handle_data(context, data):
    context.i += 1

    long_window = 60
    minutes = 1440
    minutes_str= str(minutes) + 'T'
    # minutes_str = '60T'

    now = data.current_dt

    if context.i < long_window*minutes:
        return

    if not context.i % minutes:
        hist_data = context.alldata.loc[now - pd.to_timedelta(np.arange(long_window * minutes),'T')].sort_index()[:-1]
        resample_hist_data = hist_data.resample(minutes_str).agg({'price': 'mean', 'volume': 'sum', 'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})



        #Compute Technical Indicators
        # ema_short= resample_hist_data.price.ewm(span=12).mean()
        ema_short = ta.EMA(resample_hist_data.price.values, 12)
        ema_long  = ta.EMA(resample_hist_data.price.values, 20)
        macd, signal, hist = ta.MACD(resample_hist_data.price.values, 26, 12, 9)
        adx = ta.ADX(resample_hist_data.high.values, resample_hist_data.low.values, resample_hist_data.close.values, timeperiod=14)
        mfi = ta.MFI(resample_hist_data.high.values, resample_hist_data.low.values, resample_hist_data.close.values, resample_hist_data.volume.values, timeperiod=14)[-1]

        pdi = ta.PLUS_DI(resample_hist_data.high.values, resample_hist_data.low.values, resample_hist_data.close.values,timeperiod=14)
        mdi = ta.MINUS_DI(resample_hist_data.high.values, resample_hist_data.low.values, resample_hist_data.close.values,timeperiod=14)

        price = context.alldata.loc[now].price

        if context.base_price is None:
            context.base_price = price
        price_change = (price - context.base_price) / context.base_price





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

        set_trailing_stop(context, data)


        if  mfi < 25:
            context.oversold_state = 1
            context.oversold_time = now

        if (macd[-1] < signal[-1] and macd[-2]>signal[-2]):
            context.lastcrossover_time = now



        oversold_duration = (now - context.oversold_time).total_seconds()/60/60
        lastcross_duration = (now - context.lastcrossover_time).total_seconds()/60/60





        # buy_signal  = ((context.oversold_state==1 and oversold_duration > 0 and oversold_duration <= 24) and (
        lastcross_signal = lastcross_duration > 24
        oversold_signal = context.oversold_state==1 and oversold_duration > 0 and oversold_duration <= 24
        momentum_signal = (np.mean(adx[-5:]>36)) or (np.mean(adx[-5:])>30 and pdi[-1] > mdi[-1] and pdi[-2] > mdi[-2] and pdi[-3] > mdi[-3])
        macd_signal = (macd[-1] > signal[-1]) and (macd[-2] < signal[-2])
        #buy_signal = lastcross_signal and ((oversold_signal and price < ema_long[-1]) or price>ema_long[-1])  and momentum_signal and macd_signal
        buy_signal = macd_signal and momentum_signal
        #sell_signal = (price > context.portfolio.positions[context.asset].cost_basis*context.profit_taking_pct and pos_amount > 0) or (price < context.stop_price) or (price >context.portfolio.positions[context.asset].cost_basis and price<context.trailing_stop_price)
        sell_signal = (macd[-1] < signal[-1]) and (macd[-2] > signal[-2])
        record(price=price,
               cash=context.portfolio.cash,
               amount = pos_amount,
               cost = context.portfolio.positions[context.asset].cost_basis,
               stop_price = context.stop_price,
               trailing_stop_price = context.trailing_stop_price,
               price_change=price_change,
               ema_short=ema_short[-1],
               ema_long=ema_long[-1],
               macd=macd[-1],
               signal=signal[-1],
               hist=hist[-1]/price*100,
               adx=adx[-1],
               mfi=mfi,
               pdi=pdi[-1],
               mdi=mdi[-1]
               )




        if buy_signal and pos_amount == 0:
            # we buy 100% of our portfolio for this asset
            order_target_percent(context.asset, 1)
            context.oversold_state = 0

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
    ax1 = plt.subplot(611)
    perf.loc[:, ['adx','pdi','mdi']].plot(ax=ax1,label=['ADX','PDI','MDI'])
    ax1.set_ylabel('ADX\n({})'.format(base_currency))
    ax1.axhline(30,color='g')
    start, end = ax1.get_ylim()
    ax1.yaxis.set_ticks(np.arange(start, end, (end - start) / 5))

    # Second chart: Plot asset price, moving averages and buys/sells
    ax2 = plt.subplot(612, sharex=ax1)
    perf.loc[:, ['price','ema_long']].plot(
        ax=ax2,
        label=['Price','EMA20'])

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
    ax3 = plt.subplot(613, sharex=ax1)
    perf.loc[:, ['algorithm_period_return', 'price_change']].plot(ax=ax3,
                                                                  label=['Algo Return', 'HODL Return'])
    ax3.set_ylabel('Percent Change')
    start, end = ax3.get_ylim()
    ax3.yaxis.set_ticks(np.arange(start, end, (end - start) / 5))

    # Fourth chart: Plot our indicator
    ax4 = plt.subplot(614, sharex=ax1)
    perf.loc[:, ['macd', 'signal']].plot(ax=ax4)
    ax4.set_ylabel('macd\n({})'.format(base_currency))
    start, end = ax4.get_ylim()
    ax4.yaxis.set_ticks(np.arange(0, end, end / 5))

    # Fourth chart: Plot our indicator
    ax5 = plt.subplot(615, sharex=ax1)
    perf.loc[:, ['mfi']].plot(ax=ax5)
    ax5.set_ylabel('Money Flow Index\n({})'.format(base_currency))
    start, end = ax5.get_ylim()
    ax5.yaxis.set_ticks(np.arange(0, end, end / 5))
    ax5.axhline(25, color='r')
    ax5.axhline(80, color='g')

    # Fourth chart: Plot our indicator
    ax6 = plt.subplot(616, sharex=ax1)
    perf.loc[:, ['hist']].plot(ax=ax6)
    ax6.set_ylabel('Hist Pct\n({})'.format(base_currency))
    start, end = ax6.get_ylim()
    ax6.yaxis.set_ticks(np.arange(0, end, end / 5))
    # ax6.axhline(25, color='r')
    # ax6.axhline(80, color='g')

    plt.show()
    perf.to_csv(DATA_PATH+"macd_test.csv")



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
        start=pd.to_datetime('2017-01-01 00:00:00', utc=True),
        end=pd.to_datetime('2017-05-10 00:00:00', utc=True),
    )