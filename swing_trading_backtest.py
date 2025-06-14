import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# === LOAD DATA ===
# CSV must have: TckrSymb, FinInstrmNm, date, open, high, low, close, volume
dtype_dict = {
    'TckrSymb': str,
    'date': str,
    'open': 'float32',
    'high': 'float32',
    'low': 'float32',
    'close': 'float32',
    'volume': 'float32'
}

chunk_size = 10000
all_chunks = []

for chunk in pd.read_csv("historical_stock_data.csv", 
                         usecols=dtype_dict.keys(),
                         dtype=dtype_dict,
                         parse_dates=["date"],
                         chunksize=chunk_size):
    all_chunks.append(chunk)

df = pd.concat(all_chunks)
df.sort_values(by=["TckrSymb", "date"], inplace=True)

# === PARAMETERS ===
sequence_length = 60
indicator_cols = ['RSI', 'MACD', 'BB_bbm', 'OBV', 'ATR']
raw_cols = ['open', 'high', 'low', 'close', 'volume']
feature_cols = raw_cols + indicator_cols

# === BACKTESTING FUNCTION FOR A SINGLE STOCK ===
def run_backtest(symbol, stock_data, sequence_length, feature_cols, indicator_cols):
    print(f"Processing stock: {symbol}")
    stock = stock_data[stock_data['TckrSymb'] == symbol].copy()

    # === SKIP SHORT STOCK HISTORY ===
    if len(stock) < sequence_length + 1:
        return None

    # === INDICATOR CALCULATION ===
    stock['RSI'] = RSIIndicator(close=stock['close']).rsi()
    stock['MACD'] = MACD(close=stock['close']).macd_diff()
    stock['BB_bbm'] = BollingerBands(close=stock['close']).bollinger_mavg()
    stock['OBV'] = OnBalanceVolumeIndicator(close=stock['close'], volume=stock['volume']).on_balance_volume()
    
    try:
        stock['ATR'] = AverageTrueRange(high=stock['high'], low=stock['low'], close=stock['close']).average_true_range()
    except IndexError:
        stock['ATR'] = np.nan 

    # === FILL MISSING VALUES ===
    stock[indicator_cols] = stock[indicator_cols].bfill().ffill()

    # === WEIGHTED SIGNAL SCORE (equal weight for each) ===
    stock['signal_score'] = (
        0.2 * stock['RSI'] +
        0.2 * stock['MACD'] +
        0.2 * stock['BB_bbm'] +
        0.2 * stock['OBV'] +
        0.2 * stock['ATR']
    )

    # === LABEL GENERATION ===
    # Buy signal if signal score > 0.75
    stock['buy_signal'] = (stock['signal_score'] > 0.75).astype(int)
    # Sell signal if RSI is above 70 (example exit strategy)
    stock['sell_signal'] = (stock['RSI'] > 70).astype(int)


    # === BASIC BACKTESTING LOGIC ===
    initial_capital = 100000
    position = 0
    buy_price = 0
    buy_date = None
    equity_curve = [initial_capital]
    trades = []

    for i in range(sequence_length, len(stock)):
        current_price = stock['close'].iloc[i]
        buy_signal = stock['buy_signal'].iloc[i]
        sell_signal = stock['sell_signal'].iloc[i]

        if buy_signal == 1 and position == 0:  # Buy signal
            position = initial_capital / current_price
            buy_price = current_price
            buy_date = stock['date'].iloc[i]
            equity_curve.append(initial_capital) # Equity remains the same until sale
        elif sell_signal == 1 and position > 0:  # Sell signal
            sell_price = current_price
            sell_date = stock['date'].iloc[i]
            pnl = (sell_price - buy_price) * position
            initial_capital += pnl
            trades.append({
                'buy_date': buy_date.strftime('%Y-%m-%d') if hasattr(buy_date, 'strftime') else str(buy_date),
                'buy_price': buy_price,
                'sell_date': sell_date.strftime('%Y-%m-%d') if hasattr(sell_date, 'strftime') else str(sell_date),
                'sell_price': sell_price,
                'pnl': pnl
            })
            position = 0
            buy_date = None # Reset buy_date after selling
            equity_curve.append(initial_capital)
        else:
            # Update equity curve to reflect current value of holdings
            if position > 0:
                equity_curve.append(initial_capital + (current_price - buy_price) * position)
            else:
                equity_curve.append(initial_capital) # No trade, equity remains the same

    # If position is still open at the end, close it
    if position > 0:
        sell_price = stock['close'].iloc[-1]
        sell_date = stock['date'].iloc[-1]
        pnl = (sell_price - buy_price) * position
        initial_capital += pnl
        trades.append({
            'buy_date': str(buy_date.date()) if hasattr(buy_date, 'date') else str(buy_date),
            'buy_price': buy_price,
            'sell_date': str(sell_date.date()) if hasattr(sell_date, 'date') else str(sell_date),
            'sell_price': sell_price,
            'pnl': pnl
        })
        equity_curve.append(initial_capital)


    # === CALCULATE BACKTEST METRICS ===
    total_pnl = sum([trade['pnl'] for trade in trades])
    max_drawdown = 0
    peak = equity_curve[0]
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        drawdown = (peak - equity) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    total_trades = len(trades)
    profitable_trades = sum([1 for trade in trades if trade['pnl'] > 0])
    profit_factor = (
        sum([trade['pnl'] for trade in trades if trade['pnl'] > 0]) /
        abs(sum([trade['pnl'] for trade in trades if trade['pnl'] < 0]))
        if sum([trade['pnl'] for trade in trades if trade['pnl'] < 0]) != 0 else np.inf
    )

    # === CALCULATE DRAWDOWN ===
    peak = equity_curve[0]
    drawdown = []
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        drawdown.append((peak - equity) / peak)

    # === CALCULATE RISK/PERFORMANCE RATIOS ===
    # Assuming daily data for annualization (252 trading days)
    trading_days_in_year = 252
    returns = pd.Series(equity_curve).pct_change().dropna()

    # Sharpe Ratio
    # Assuming risk-free rate is 0 for simplicity
    sharpe_ratio = np.sqrt(trading_days_in_year) * returns.mean() / returns.std() if returns.std() != 0 else np.nan

    # Sortino Ratio
    # Assuming risk-free rate is 0 for simplicity
    downside_returns = returns[returns < 0]
    sortino_ratio = np.sqrt(trading_days_in_year) * returns.mean() / downside_returns.std() if downside_returns.std() != 0 else np.nan


    return {
        'symbol': symbol,
        'total_pnl': total_pnl,
        'max_drawdown': max_drawdown,
        'total_trades': total_trades,
        'profitable_trades': profitable_trades,
        'profit_factor': profit_factor,
        'equity_curve': equity_curve,
        'drawdown': drawdown,
        'trades': trades,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'buy_signals': stock['buy_signal'].tolist(),
        'sell_signals': stock['sell_signal'].tolist()
    }


# === MULTIPROCESSING ===
if __name__ == '__main__':
    # Filter for a single equity for faster execution
    symbols_to_backtest = ['TCS'] # You can change 'TCS' to any other symbol

    backtest_results = []

    # Process only the selected symbols
    df_filtered = df[df['TckrSymb'].isin(symbols_to_backtest)]

    with Pool(cpu_count()) as p:
        results = list(tqdm(p.starmap(run_backtest, [(symbol, df_filtered, sequence_length, feature_cols, indicator_cols) for symbol in symbols_to_backtest]), total=len(symbols_to_backtest), desc="Running Backtests"))

    # Filter out None results (for stocks with insufficient data)
    backtest_results = [result for result in results if result is not None]

    # === AGGREGATE AND PRINT RESULTS ===
    print("\n=== Backtest Results ===")
    for result in backtest_results:
        print(f"\nSymbol: {result['symbol']}")
        print(f"Total P&L: {result['total_pnl']:.2f}")
        print(f"Max Drawdown: {result['max_drawdown']:.2f}")
        print(f"Total Trades: {result['total_trades']}")
        print(f"Profitable Trades: {result['profitable_trades']}")
        print(f"Profit Factor: {result['profit_factor']:.2f}")
        # You can add more detailed trade information or equity curve plotting here
