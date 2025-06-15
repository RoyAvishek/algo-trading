import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, SMAIndicator, ADXIndicator
from ta.volatility import AverageTrueRange
from ta.volume import ChaikinMoneyFlowIndicator
import warnings
warnings.filterwarnings('ignore')

class SwingTradingStrategy:
    def __init__(self):
        # Strategy parameters can be added here if needed in the future
        pass

    def generate_signals(self, stock_data):
        """
        Generates trading signals for the given stock data.
        """
        stock = stock_data.copy()

        if len(stock) < 200: # Need enough data for SMA200
            return stock # Or handle as an error/warning

        stock = stock.sort_values('date').reset_index(drop=True)

        # Calculate indicators
        stock['RSI'] = RSIIndicator(close=stock['close'], window=14).rsi()
        stock['EMA20'] = EMAIndicator(close=stock['close'], window=20).ema_indicator()
        stock['EMA50'] = EMAIndicator(close=stock['close'], window=50).ema_indicator()
        stock['SMA200'] = SMAIndicator(close=stock['close'], window=200).sma_indicator()
        stock['ATR'] = AverageTrueRange(high=stock['high'], low=stock['low'], close=stock['close'], window=14).average_true_range()
        stock['CMF'] = ChaikinMoneyFlowIndicator(high=stock['high'], low=stock['low'], close=stock['close'], volume=stock['volume'], window=20).chaikin_money_flow()
        stock['ADX'] = ADXIndicator(high=stock['high'], low=stock['low'], close=stock['close'], window=14).adx()
        stock['volume_ma'] = stock['volume'].rolling(window=20).mean()

        # Generate entry signal
        stock['entry_signal'] = 0
        for i in range(len(stock)):
            if stock['close'].iloc[i] > stock['EMA20'].iloc[i] and stock['volume'].iloc[i] > stock['volume_ma'].iloc[i] * 1.2:
                if 30 < stock['RSI'].iloc[i] < 80 and stock['ADX'].iloc[i] > 20:
                    stock.loc[i, 'entry_signal'] = 1

        return stock
