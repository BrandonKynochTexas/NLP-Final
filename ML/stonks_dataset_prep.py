# https://python-tradingview-ta.readthedocs.io/en/latest/overview.html
# https://github.com/StreamAlpha/tvdatafeed

from tradingview_ta import TA_Handler, Interval, Exchange
from tvDatafeed import TvDatafeed
from tvDatafeed import Interval as tvDFI
import tv_datafeed_utils as tvd_utils

import stonks_strategy as strategy

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.markers as plt_markers

import datetime
import pytz
from tzlocal import get_localzone


current_timezone = get_localzone()
nasdaq_timezone = pytz.timezone("America/New_York")

# TODO:
#   Pad TV_bars before reshaping to day format so that previous days have correct ranges
#
#   Log into trading view properly using google account so that we can access
#   very short time intervals   



class StockData:
    def __init__(
            self,
            ticker,
            exchange='NASDAQ',
            interval=tvDFI.in_15_minute,
            num_days=1) -> None:
        self.ticker = ticker
        self.exchange = exchange
        self.interval = interval
        self.num_days = num_days

        # self.tv_dataset = TVConnection(ticker=ticker, exchange=exchange, interval=interval)
        # self.bars = self.tv_dataset.get_dataset(num_days=num_days, show_plot=True)
        # self.analysis = self.tv_dataset.get_analysis()

        self.article_collector = ArticleCollector()



# 6 hours 30 minutes in regular trading hours (390 minutes)
# Without login: 1 minute interval will return at most 15 days of data (5824 bars)

# Trading View Connection class
class TVConnection:
    # tv_username = "*******@gmail.com"
    # tv_password = ""
    # tv_df = TvDatafeed(tv_username, tv_password)


    def __init__(
            self,
            ticker,
            exchange='NASDAQ',
            interval=tvDFI.in_15_minute,
            ):

        self.ticker = ticker
        self.exchange = exchange
        self.interval = interval

        self.tv_df = TvDatafeed()

        self.handler = TA_Handler(
            symbol=self.ticker,
            screener="america",
            exchange=self.exchange,
            interval=self.interval
        )
    

    def get_dataset(
            self,
            num_days=1,
            show_plot=False):

        bars_per_day = int(math.ceil(tvd_utils.bars_per_day(self.interval.value)))
        bars = num_days * bars_per_day

        if bars <= 10:
            print('Invalid call to \'get_dataset\': too few bars')
            return

        # returns:
        #   ['NASDAQ:NVDA' 153.15 153.3 153.04 153.04 6003.0]
        bars = self.tv_df.get_hist(
            symbol=self.ticker,
            exchange=self.exchange,
            interval=self.interval,
            n_bars=bars)
        
        if type(bars) == type(None):
            print("Network error when retreiving bar data")
            return

        # Column 4 = current price
        times = np.array([current_timezone.localize(bar).timestamp() for bar in bars.index])
        bars = bars.to_numpy()[:, 4].view(dtype=object)
        bars = np.dstack((times, bars))[0]

        bars = bars.reshape((num_days, -1, 2))
        bars = np.lib.pad(bars, ((0, 0), (0, 0), (0, 1)), constant_values=(0)) # add one column to mark where to buy & sell

        for day in range(bars.shape[0]):
            bars[day] = strategy.strategy(bars[day])

            if show_plot:
                x_axis_display = np.array([
                    datetime.datetime.fromtimestamp(timestamp, tz=nasdaq_timezone).strftime('%m-%d %H:%M:%S') for timestamp in bars[day][:, 0]
                ])
                bars_plt = bars[day].copy()
                bars_plt[:, 0] = x_axis_display
                plt.plot(bars_plt[:, 0], bars_plt[:, 1])

                markers_up = np.array([bar for bar in bars_plt if bar[2] == 1])
                if len(markers_up) > 0:
                    plt.plot(markers_up[:, 0], markers_up[:, 1], marker=6, ls='none')

                markers_down = np.array([bar for bar in bars_plt if bar[2] == -1])
                if len(markers_down) > 0:
                    plt.plot(markers_down[:, 0], markers_down[:, 1], marker=7, ls='none')

                plt.title(f'{self.exchange}:{self.ticker}  -  Interval:{self.interval.value}')
                plt.show()

        print(f'Fetched TV bars:\n\t{self.interval}\n\t{bars_per_day} bars per day with {num_days} days')

        return bars

    def get_analysis(self):
        return self.handler.get_analysis().summary


class ArticleCollector:
    def __init__(self) -> None:
        pass


if __name__=="__main__":
    
    # dprep.get_dataset(ticker='VICI', exchange='NYSE')
    # dprep.get_dataset(ticker='VICI', exchange='NYSE', show_plot=True)
    # dprep.get_dataset(ticker='QCOM', show_plot=True)

    data = StockData(ticker='NVDA', interval=tvDFI.in_5_minute, num_days=6)
