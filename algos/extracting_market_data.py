
import csv
import pytz
from datetime import datetime

from catalyst.api import record, symbol, symbols
from catalyst.utils.run_algo import run_algorithm


start = datetime(2016, 1, 1, 0, 0, 0, 0, pytz.utc)
end = datetime(2018, 7, 1, 0, 0, 0, 0, pytz.utc)

EXCHANGE = "bitfinex" #bitfinex,bitmex
FREQUENCY = "minute"  #minute

DATA_PATH = "/Users/taoranli/Documents/python/cryptotrading/enigma-catalyst/data/"

FILE_NAME = DATA_PATH + EXCHANGE + "_" + FREQUENCY + "_" + start.strftime("%Y-%m-%d") + "_" + end.strftime("%Y-%m-%d")

def initialize(context):
    # Portfolio assets list
    context.asset     = symbol('btc_usd') # Bitcoin on Poloniex

    # Creates a .CSV file with the same name as this script to store results
    context.csvfile   = open(FILE_NAME +'.csv', 'w+')
    context.csvwriter = csv.writer(context.csvfile)

    context.csvwriter.writerow(["Time", "price", "volume","high", "low","open","close"])

def handle_data(context, data):
    # Variables to record for a given asset: price and volume
    # Other options include 'open', 'high', 'open', 'close'
    # Please note that 'price' equals 'close'
    date   = context.blotter.current_dt     # current time in each iteration
    price  = data.current(context.asset, 'price')
    volume = data.current(context.asset, 'volume')
    high = data.current(context.asset, 'high')
    low = data.current(context.asset, 'low')
    open = data.current(context.asset, 'open')
    close = data.current(context.asset, 'close')

    # Writes one line to CSV on each iteration with the chosen variables
    context.csvwriter.writerow([date, price, volume, high, low, open, close])

def analyze(context=None, results=None):
    # Close open file properly at the end
    context.csvfile.close()


    # Bitcoin data is available from 2015-3-2. Dates vary for other tokens.

results = run_algorithm(initialize=initialize,
                            handle_data=handle_data,
                            analyze=analyze,
                            start=start,
                            end=end,
                            exchange_name=EXCHANGE,
                            data_frequency=FREQUENCY,
                            base_currency ='usd',
                            capital_base=10000 )