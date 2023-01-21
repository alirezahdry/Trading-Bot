import talib as ta
import pandas as pd
import pytrendline
from typing import Optional, Tuple, Dict, Any
import os
import time


class MyStrategy():
    @staticmethod
    def version(self) -> str:
        """
        Returns version of the strategy.
        """
        return "1.1"

    def __init__(self):
        super().__init__()
        self.candles_df = None
        self.resistance_trendlines = None
        self.support_trendlines = None
        self.stop_loss = None
        self.take_profit_1 = None
        self.take_profit_2 = None
        self.take_profit_3 = None

        """

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy
    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_entry_trend, populate_exit_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Can this strategy go short?
    can_short: bool = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    # stoploss = -0.10

    # Trailing stoploss
    # trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Optimal timeframe for the strategy.
    timeframe = '15m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    # To disable ROI completely, set it to an insanely high number:
    minimal_roi = {
        "0": 100
    }

    def informative_pairs(self):

        #         def informative_pairs(self):
        #     return [
        #         ("ETH/USDT", "5m", ""),  # Uses default candletype, depends on trading_mode (recommended)
        #         ("ETH/USDT", "5m", "spot"),  # Forces usage of spot candles (only valid for bots on spot markets).
        #         ("BTC/TUSD", "15m", "futures"),  # Uses futures candles (only bots with `trading_mode=futures`)
        #         ("BTC/TUSD", "15m", "mark"),  # Uses mark candles (only bots with `trading_mode=futures`)
        #     ]

        # get access to all pairs available in whitelist.
        pairs = self.dp.current_whitelist()
        # Assign tf to each pair, so they can be downloaded and cached for strategy.
        informative_pairs = [(pair, '1d') for pair in pairs]
        # Optionally Add additional "static" pairs
        informative_pairs += [("ETH/USDT", "15m"),
                              ("BTC/USDT", "15m"),
                              ]
        return informative_pairs

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        inf_tf = '15m'
        # Get the informative pair
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf)
        # Get the 14 day rsi
        informative['rsi'] = ta.RSI(informative, timeperiod=14)
        # Use the helper function merge_informative_pair to safely merge the pair
        # Automatically renames the columns and merges a shorter timeframe dataframe and a longer timeframe informative pair
        # use ffill to have the 1d value available in every row throughout the day.
        # Without this, comparisons between columns of the original and the informative pair would only work once per day.
        # Full documentation of this method, see below
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)

        ##############################################################        
        self.candles_df = dataframe
        # detect support and resistance trendlines
        candlestick_data = pytrendline.CandlestickData(
            df=self.candles_df,
            time_interval="15m",  # choose between 1m,3m,5m,10m,15m,30m,1h,1d
            open_col="Open",  # name of the column containing candle "Open" price
            high_col="High",  # name of the column containing candle "High" price
            low_col="Low",  # name of the column containing candle "Low" price
            close_col="Close",  # name of the column containing candle "Close" price
            datetime_col="Date"
            # name of the column containing candle datetime price (use none if datetime is in index)
        )

        results = pytrendline.detect(
            candlestick_data=candlestick_data,
            trend_type=pytrendline.TrendlineTypes.BOTH,
            first_pt_must_be_pivot=False,
            last_pt_must_be_pivot=False,
            all_pts_must_be_pivots=False,
            trendline_must_include_global_maxmin_pt=False,
            min_points_required=3,
            scan_from_date=None,
            ignore_breakouts=True,
            config={}
        )
        self.resistance_trendlines = results['resistance_trendlines']
        self.support_trendlines = results['support_trendlines']

        # Add RSI indicator to the dataframe
        dataframe['RSI'] = ta.RSI(dataframe['Close'], timeperiod=14)
        # check for divergence or convergence for resistance trendlines
        condition = []
        for index, row in self.resistance_trendlines.iterrows():
            trendline_points = row['pointset_indeces']
            # Extract RSI at the beginning and end points of the trend line
            rsi_start = dataframe.loc[trendline_points[0], 'RSI']
            rsi_end = dataframe.loc[trendline_points[-1], 'RSI']
            # calculate the slope of the RSI trendline
            x1, x2 = trendline_points[0], trendline_points[-1]
            y1, y2 = rsi_start, rsi_end
            rsi_slope = (y2 - y1) / (x2 - x1)
            # check for divergence or convergence
            if rsi_slope * row['slope'] < 0:
                condition.append("Divergence")
            elif rsi_slope * row['slope'] > 0:
                condition.append("Convergence")
            else:
                condition.append("Indeterminate")
            self.resistance_trendlines['condition'] = condition
            # check for divergence or convergence for support trendlines
            condition = []
            for index, row in self.support_trendlines.iterrows():
                trendline_points = row['pointset_indeces']

                # Extract RSI at the beginning and end points of the trend line
                rsi_start = dataframe.loc[trendline_points[0], 'RSI']
                rsi_end = dataframe.loc[trendline_points[-1], 'RSI']

                # calculate the slope of the RSI trendline
                x1, x2 = trendline_points[0], trendline_points[-1]
                y1, y2 = rsi_start, rsi_end
                rsi_slope = (y2 - y1) / (x2 - x1)

                # check for divergence or convergence
                if rsi_slope * row['slope'] < 0:
                    condition.append("Divergence")
                elif rsi_slope * row['slope'] > 0:
                    condition.append("Convergence")
                else:
                    condition.append("Indeterminate")
            self.support_trendlines['condition'] = condition
            return dataframe

    def identify_candlestick_pattern(self, trendline_endpoint: int, next_candle: int) -> str:
        """
        Method that identifies the candlestick pattern at the endpoint and the next candle of a support trendline

        :param trendline_endpoint: index of the endpoint of the trendline
        :param next_candle: index of the next candle
        :return: string representation of the candlestick pattern
        """

        # Code to identify candlestick pattern goes here

        pattern_list = ta.get_function_groups()['Pattern Recognition']
        pattern_list = [x.replace("CDL", "") for x in pattern_list]

        # Identify the candlestick pattern at the endpoint of a resistance trendline
        resistance_trendlines['pattern'] = None
        for index, row in resistance_trendlines.iterrows():
            trendline_end = row['ends_at_index']
            open_price = candles_df.loc[trendline_end:, 'Open'].values
            high_price = candles_df.loc[trendline_end:, 'High'].values
            low_price = candles_df.loc[trendline_end:, 'Low'].values
            close_price = candles_df.loc[trendline_end:, 'Close'].values
            res = {}
            for pattern in pattern_list:
                res[pattern] = getattr(ta, 'CDL' + pattern)(open_price, high_price, low_price, close_price)[-1]
            pattern = max(res, key=res.get)
            resistance_trendlines.at[index, 'pattern'] = pattern

        # Identify the candlestick pattern at the endpoint of a support trendline
        support_trendlines['pattern'] = None
        for index, row in support_trendlines.iterrows():
            trendline_end = row['ends_at_index']
            open_price = candles_df.loc[trendline_end:, 'Open'].values
            high_price = candles_df.loc[trendline_end:, 'High'].values
            low_price = candles_df.loc[trendline_end:, 'Low'].values
            close_price = candles_df.loc[trendline_end:, 'Close'].values
            res = {}
            for pattern in pattern_list:
                res[pattern] = getattr(ta, 'CDL' + pattern)(open_price, high_price, low_price, close_price)[-1]
            pattern = max(res, key=res.get)
            support_trendlines.at[index, 'pattern'] = pattern

        # Identify the candlestick pattern at the endpoint of a resistance trendline
        resistance_trendlines['next_candle_pattern'] = None
        for index, row in resistance_trendlines.iterrows():
            trendline_end = row['ends_at_index']
            if trendline_end + 1 > data.index[-1]:
                resistance_trendlines.at[index, 'next_candle_pattern'] = "NOT CLOSE YET"
            else:
                open_price = np.array([candles_df.loc[trendline_end + 1, 'Open']])
                high_price = np.array([candles_df.loc[trendline_end + 1, 'High']])
                low_price = np.array([candles_df.loc[trendline_end + 1, 'Low']])
                close_price = np.array([candles_df.loc[trendline_end + 1, 'Close']])
            res = {}
            for pattern in pattern_list:
                res[pattern] = getattr(ta, 'CDL' + pattern)(open_price, high_price, low_price, close_price)[-1]
            pattern = max(res, key=res.get)
            resistance_trendlines.at[index, 'next_candle_pattern'] = pattern

            # Identify the candlestick pattern at the endpoint of a support trendline
        support_trendlines['next_candle_pattern'] = None
        for index, row in support_trendlines.iterrows():
            trendline_end = row['ends_at_index']
            if trendline_end + 1 > data.index[-1]:
                support_trendlines.at[index, 'next_candle_pattern'] = "NOT CLOSE YET"
            else:
                open_price = np.array([candles_df.loc[trendline_end + 1, 'Open']])
                high_price = np.array([candles_df.loc[trendline_end + 1, 'High']])
                low_price = np.array([candles_df.loc[trendline_end + 1, 'Low']])
                close_price = np.array([candles_df.loc[trendline_end + 1, 'Close']])
            res = {}
            for pattern in pattern_list:
                res[pattern] = getattr(ta, 'CDL' + pattern)(open_price, high_price, low_price, close_price)[-1]
            pattern = max(res, key=res.get)
            support_trendlines.at[index, 'next_candle_pattern'] = pattern

        return candlestick_pattern

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """

        for index, row in self.support_trendlines.iterrows():
            trendline_endpoint = row['pointset_indeces'][-1]
            next_candle = trendline_endpoint + 1
            candlestick_pattern = self.identify_candlestick_pattern(trendline_endpoint, next_candle)
            if (row['condition'] == "Convergence") and (candlestick_pattern == "Bullish reversal"):
                dataframe.loc[trendline_endpoint, ['enter_long', 'enter_tag']] = (1, 'Convergence')

        for index, row in self.resistance_trendlines.iterrows():
            trendline_endpoint = row['pointset_indeces'][-1]
            next_candle = trendline_endpoint + 1
            candlestick_pattern = self.identify_candlestick_pattern(trendline_endpoint, next_candle)
            if (row['condition'] == "Divergence") and (candlestick_pattern == "Bearish reversal"):
                dataframe.loc[trendline_endpoint, ['enter_short', 'enter_tag']] = (1, 'Divergence')
        return dataframe

    use_custom_stoploss = True

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_profit: float, **kwargs) -> tuple[bool, None] | tuple[bool, dict[str, Any | None]] | \
                                                            tuple[bool, dict[str, float | int | Any]]:
        atr = self.dataframe.atr(pair=pair, period=14)

        if trade.is_long:
            stoploss_price = (order["Price"] - atr)
            current_profit_1 = order["Price"] + atr
            current_profit_2 = order["Price"] + atr * 3
            current_profit_3 = order["Price"] + atr * 6
            amount = trade.amount
            if current_price >= current_profit_1:
                stoploss_price = order["Price"]
                current_profit = current_profit_2
                trade.close(amount=trade.amount * 0.5)
            elif current_price >= current_profit_2:
                stoploss_price = current_profit_1
                current_profit = current_profit_3
                trade.close(amount=trade.amount * 0.25)
            elif current_price >= take_profit_3:
                trade.close(amount=trade.amount * 0.25)
        else:  # trade.is_short
            stoploss_price = order["Price"] + atr
            current_profit_1 = order["Price"] - atr
            current_profit_2 = order["Price"] - atr * 3
            current_profit_3 = order["Price"] - atr * 6
            amount = trade.amount
            if current_price <= current_profit_1:
                stoploss_price = order["Price"]
                current_profit = current_profit_2
                trade.close(amount=trade.amount * 0.5)
            elif current_price <= current_profit_2:
                stoploss_price = current_profit_1
                current_profit = current_profit_3
                trade.close(amount=trade.amount * 0.25)
            elif current_price <= take_profit_3:
                trade.close(amount=trade.amount * 0.25)
        # If stoploss and current profit are not defined, return False
        if stoploss_price is None and current_profit is None:
            return False, None
        else:
            # If the current rate is equal to the stop loss, close the position
            if stoploss_price == current_price:
                return True, {"stop_loss": stoploss_price, "current_profit": None, "amount": trade.amount}
            else:
                return True, {"stop_loss": stoploss_price, "current_profit": current_profit, "amount": amount}

    def check_buy_timeout(self, pair: str, trade: Trade, order: dict, **kwargs) -> bool:
        # For long positions, check if current price is higher than the buy price + some threshold
        if trade.is_long:
            ob = self.dp.orderbook(pair, 1)
            current_price = ob['bids'][0][0]
            threshold = 0.005 # 0.5% threshold
            if current_price > order['price'] * (1 + threshold):
                return True
        # For short positions, check if current price is lower than the buy price - some threshold
        else:
            ob = self.dp.orderbook(pair, 1)
            current_price = ob['asks'][0][0]
            threshold = 0.005 # 0.5% threshold
            if current_price < order['price'] * (1 - threshold):
                return True
        return False

    def check_sell_timeout(self, pair: str, trade: Trade, order: dict, **kwargs) -> bool:
        # For long positions, check if current price is lower than the sell price - some threshold
        if trade.is_long:
            ob = self.dp.orderbook(pair, 1)
            current_price = ob['asks'][0][0]
            threshold = 0.005 # 0.5% threshold
            if current_price < order['price'] * (1 - threshold):
                return True
        # For short positions, check if current price is higher than the sell price + some threshold
        else:
            ob = self.dp.orderbook(pair, 1)
            current_price = ob['bids'][0][0]
            threshold = 0.005 # 0.5% threshold
            if current_price > order['price'] * (1 + threshold):
                return True
        return False

